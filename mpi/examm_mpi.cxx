#include <chrono>
#include <iomanip>
using std::fixed;
using std::setprecision;
using std::setw;

#include <cstdint>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <sstream>

#include <mutex>
using std::mutex;

#include <string>
using std::string;

#include <thread>
using std::thread;

#include <vector>
using std::vector;

#include "common/log.hxx"
#include "common/process_arguments.hxx"
#include "examm/examm.hxx"
#include "mpi.h"
#include "rnn/generate_nn.hxx"
#include "time_series/time_series.hxx"
#include "weights/weight_rules.hxx"
#include "weights/weight_update.hxx"

#define WORK_REQUEST_TAG  1
#define GENOME_LENGTH_TAG 2
#define GENOME_TAG        3
#define TERMINATE_TAG     4

// P2P tags (ring-based migration/backup + termination consensus).
// We keep the legacy tags above for backward compatibility, but the new
// decentralized peer loop uses the following tags instead.
#define MIGRATE_GENOME_TAG  5
#define BACKUP_GENOME_TAG   6
#define MIGRATE_GENOME_DATA_TAG (MIGRATE_GENOME_TAG + 100)
#define BACKUP_GENOME_DATA_TAG  (BACKUP_GENOME_TAG + 100)
#define TERMINATION_TOKEN_TAG 7

// Seed broadcast (only used during startup).
#define SEED_GENOME_LENGTH_TAG   8
#define SEED_GENOME_DATA_TAG     9

// Fault-tolerance tags.
// Heartbeat: each rank pings its physical ring successor every N ms.
// Silence beyond the timeout causes the successor to declare PEER_FAILED.
#define HEARTBEAT_TAG         11

// Broadcast by the detecting rank when its predecessor stops heartbeating.
// Payload: 1 x MPI_INT = failed rank id.
#define PEER_FAILED_TAG       12

// Sent by a recovering rank to the lowest active rank to request genome seed.
// Payload: 1 x MPI_INT = rejoining rank id.
#define REJOIN_REQUEST_TAG    13

// Broadcast by the rank that responds to REJOIN_REQUEST, notifying all peers.
// Payload: 1 x MPI_INT = rejoining rank id.
#define REJOIN_NOTIFY_TAG     14

// Intentional/graceful shutdown: rank broadcasts before permanently leaving.
// Payload: 1 x MPI_INT = shutting-down rank id.
#define GRACEFUL_SHUTDOWN_TAG 15

mutex examm_mutex;

vector<string> arguments;

EXAMM* examm;
WeightUpdate* weight_update_method;

bool finished = false;

vector<vector<vector<double> > > training_inputs;
vector<vector<vector<double> > > training_outputs;
vector<vector<vector<double> > > validation_inputs;
vector<vector<vector<double> > > validation_outputs;

static uint64_t stable_hash_fnv1a_64(const std::string& s) {
    // Deterministic across peers/ranks: FNV-1a 64-bit.
    uint64_t hash = 14695981039346656037ULL;
    for (unsigned char c : s) {
        hash ^= static_cast<uint64_t>(c);
        hash *= 1099511628211ULL;
    }
    return hash;
}

static int32_t genome_owner_rank(const RNN_Genome* genome, int32_t max_rank) {
    // Uses the genome's deterministic structural hash to pick an owner rank.
    std::string structural_hash = genome->get_structural_hash();
    if (structural_hash.size() == 0) {
        // Should generally already be computed by genome constructors, but be safe.
        // Note: structural_hash is computed in assign_reachability(), which is non-const.
        RNN_Genome* non_const = const_cast<RNN_Genome*>(genome);
        non_const->assign_reachability();
        structural_hash = non_const->get_structural_hash();
    }
    uint64_t h = stable_hash_fnv1a_64(structural_hash);
    return static_cast<int32_t>(h % static_cast<uint64_t>(max_rank));
}

// Returns the next alive rank after `my_rank` in the sorted active_ranks ring.
// If my_rank is the highest alive rank, wraps around to active_ranks[0].
static int32_t next_alive_rank(int32_t my_rank, const std::vector<int32_t>& active_ranks) {
    for (int32_t r : active_ranks) {
        if (r > my_rank) return r;
    }
    return active_ranks[0];  // wrap around
}

// Ownership mapping using only currently-alive ranks so MIGRATE never
// targets a dropped peer.
static int32_t genome_owner_rank_dynamic(
    const RNN_Genome* genome,
    const std::vector<int32_t>& active_ranks
) {
    std::string structural_hash = genome->get_structural_hash();
    if (structural_hash.empty()) {
        RNN_Genome* nc = const_cast<RNN_Genome*>(genome);
        nc->assign_reachability();
        structural_hash = nc->get_structural_hash();
    }
    uint64_t h   = stable_hash_fnv1a_64(structural_hash);
    int32_t  idx = static_cast<int32_t>(h % static_cast<uint64_t>(active_ranks.size()));
    return active_ranks[idx];
}

// Configuration for fault-tolerance simulation (parsed from CLI args).
struct DropoutConfig {
    // Unintentional failure: rank goes silent, peers detect via heartbeat timeout, rank recovers later.
    int32_t dropout_rank            = -1;   // -1 = disabled
    double  dropout_after_seconds   = 0.0;
    double  recovery_after_seconds  = 30.0; // seconds after dropout before rejoining

    // Intentional/graceful shutdown: rank broadcasts full state then permanently leaves.
    int32_t shutdown_rank           = -1;   // -1 = disabled
    double  shutdown_after_seconds  = 0.0;

    // Heartbeat tuning.
    int32_t heartbeat_interval_ms   = 1000; // how often each rank pings its successor
    int32_t heartbeat_timeout_ms    = 5000; // silence before successor declares failure
};

enum class GenomeTransferKind : int32_t {
    MIGRATE = 0,
    BACKUP = 1
};

struct IncomingGenomeTransfer {
    int32_t source = -1;
    GenomeTransferKind kind = GenomeTransferKind::MIGRATE;
    int32_t length = 0;
    char* buffer = nullptr;  // length + 1
    std::vector<MPI_Request> requests;
};

struct OutgoingGenomeTransfer {
    int32_t dest = -1;
    GenomeTransferKind kind = GenomeTransferKind::MIGRATE;
    int32_t length = 0;
    char* byte_array = nullptr;  // malloc'd by genome->write_to_array
    int32_t* length_ptr = nullptr;
    std::vector<MPI_Request> requests;
};

static void queue_genome_send(
    GenomeTransferKind kind,
    int32_t dest,
    RNN_Genome* genome,
    std::vector<OutgoingGenomeTransfer>& pending_outgoing
) {
    constexpr int32_t chunk_size = 32768;

    char* byte_array = nullptr;
    int32_t length = 0;
    genome->write_to_array(&byte_array, length);

    int32_t length_tag = (kind == GenomeTransferKind::MIGRATE) ? MIGRATE_GENOME_TAG : BACKUP_GENOME_TAG;
    int32_t data_tag = (kind == GenomeTransferKind::MIGRATE) ? MIGRATE_GENOME_DATA_TAG : BACKUP_GENOME_DATA_TAG;

    int32_t* length_ptr = new int32_t(length);
    OutgoingGenomeTransfer transfer;
    transfer.dest = dest;
    transfer.kind = kind;
    transfer.length = length;
    transfer.byte_array = byte_array;
    transfer.length_ptr = length_ptr;

    transfer.requests.reserve(1 + (length + chunk_size - 1) / chunk_size);

    // Length header.
    MPI_Request len_req;
    MPI_Isend(length_ptr, 1, MPI_INT, dest, length_tag, MPI_COMM_WORLD, &len_req);
    transfer.requests.push_back(len_req);

    // Chunked payload.
    int32_t offset = 0;
    while (offset < length) {
        int32_t send_size = length - offset;
        if (send_size > chunk_size) {
            send_size = chunk_size;
        }

        MPI_Request req;
        MPI_Isend(byte_array + offset, send_size, MPI_CHAR, dest, data_tag, MPI_COMM_WORLD, &req);
        transfer.requests.push_back(req);
        offset += send_size;
    }

    pending_outgoing.push_back(std::move(transfer));
}

static void post_genome_receive(
    GenomeTransferKind kind,
    int32_t source,
    int32_t length,
    std::vector<IncomingGenomeTransfer>& pending_incoming
) {
    constexpr int32_t chunk_size = 32768;

    int32_t data_tag = (kind == GenomeTransferKind::MIGRATE) ? MIGRATE_GENOME_DATA_TAG : BACKUP_GENOME_DATA_TAG;

    IncomingGenomeTransfer transfer;
    transfer.source = source;
    transfer.kind = kind;
    transfer.length = length;
    transfer.buffer = new char[length + 1];
    transfer.buffer[length] = '\0';
    transfer.requests.reserve((length + chunk_size - 1) / chunk_size);

    int32_t offset = 0;
    while (offset < length) {
        int32_t recv_size = length - offset;
        if (recv_size > chunk_size) {
            recv_size = chunk_size;
        }

        MPI_Request req;
        MPI_Irecv(transfer.buffer + offset, recv_size, MPI_CHAR, source, data_tag, MPI_COMM_WORLD, &req);
        transfer.requests.push_back(req);
        offset += recv_size;
    }

    pending_incoming.push_back(std::move(transfer));
}

static void progress_outgoing(
    std::vector<OutgoingGenomeTransfer>& pending_outgoing
) {
    for (size_t i = 0; i < pending_outgoing.size();) {
        auto& t = pending_outgoing[i];
        if (t.requests.empty()) {
            // Shouldn't happen, but handle defensively.
            if (t.byte_array) free(t.byte_array);
            if (t.length_ptr) delete t.length_ptr;
            pending_outgoing.erase(pending_outgoing.begin() + i);
            continue;
        }

        int flag = 0;
        MPI_Testall((int) t.requests.size(), t.requests.data(), &flag, MPI_STATUSES_IGNORE);
        if (flag) {
            if (t.byte_array) free(t.byte_array);
            if (t.length_ptr) delete t.length_ptr;
            pending_outgoing.erase(pending_outgoing.begin() + i);
        } else {
            i++;
        }
    }
}

static void progress_incoming(
    std::vector<IncomingGenomeTransfer>& pending_incoming,
    EXAMM* examm,
    int32_t rank,
    const std::vector<int32_t>& active_ranks
) {
    for (size_t i = 0; i < pending_incoming.size();) {
        auto& t = pending_incoming[i];
        if (t.requests.empty()) {
            delete[] t.buffer;
            pending_incoming.erase(pending_incoming.begin() + i);
            continue;
        }

        int flag = 0;
        MPI_Testall((int) t.requests.size(), t.requests.data(), &flag, MPI_STATUSES_IGNORE);
        if (!flag) {
            i++;
            continue;
        }

        // Transfer is complete.
        t.buffer[t.length] = '\0';
        RNN_Genome* genome = new RNN_Genome(t.buffer, t.length);

        bool should_insert = true;
        if (t.kind == GenomeTransferKind::MIGRATE) {
            // Use the dynamic (dropout-aware) owner so stale MIGRATE messages
            // sent before a peer dropped are still routed correctly.
            int32_t owner = active_ranks.empty()
                                ? rank
                                : genome_owner_rank_dynamic(genome, active_ranks);
            if (owner != rank) {
                should_insert = false;
            }
        }

        if (should_insert) {
            examm->inject_migrated_genome(genome);
        }

        delete genome;
        delete[] t.buffer;
        pending_incoming.erase(pending_incoming.begin() + i);
    }
}

static RNN_Genome* broadcast_genome_seed(RNN_Genome* seed_genome, int32_t rank, int32_t max_rank) {
    // Broadcast the seed genome bytes from rank 0 to all ranks.
    int32_t length = 0;
    char* byte_array = nullptr;

    if (rank == 0) {
        seed_genome->write_to_array(&byte_array, length);
    }

    MPI_Bcast(&length, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (length <= 0) {
        if (rank == 0) {
            free(byte_array);
            return seed_genome;
        }
        return nullptr;
    }

    constexpr int32_t chunk_size = 32768;

    char* recv_buffer = (rank == 0) ? nullptr : new char[length + 1];

    int32_t offset = 0;
    while (offset < length) {
        int32_t send_size = length - offset;
        if (send_size > chunk_size) {
            send_size = chunk_size;
        }

        void* ptr = nullptr;
        if (rank == 0) {
            ptr = byte_array + offset;
        } else {
            ptr = recv_buffer + offset;
        }

        MPI_Bcast(ptr, send_size, MPI_CHAR, 0, MPI_COMM_WORLD);
        offset += send_size;
    }

    if (rank == 0) {
        free(byte_array);
        return seed_genome;
    }

    recv_buffer[length] = '\0';
    RNN_Genome* received = new RNN_Genome(recv_buffer, length);
    delete[] recv_buffer;
    return received;
}

static std::vector<std::string> with_peer_output_directory(
    const std::vector<std::string>& args,
    const std::string& peer_output_directory
) {
    std::vector<std::string> out = args;
    for (size_t i = 0; i + 1 < out.size(); i++) {
        if (out[i].compare("--output_directory") == 0) {
            out[i + 1] = peer_output_directory;
            break;
        }
    }
    return out;
}

// bool random_sequence_length;
// int32_t sequence_length_lower_bound = 30;
// int32_t sequence_length_upper_bound = 100;

void send_work_request(int32_t target) {
    int32_t work_request_message[1];
    work_request_message[0] = 0;
    MPI_Send(work_request_message, 1, MPI_INT, target, WORK_REQUEST_TAG, MPI_COMM_WORLD);
}

void receive_work_request(int32_t source) {
    MPI_Status status;
    int32_t work_request_message[1];
    MPI_Recv(work_request_message, 1, MPI_INT, source, WORK_REQUEST_TAG, MPI_COMM_WORLD, &status);
}

RNN_Genome* receive_genome_from(int32_t source) {
    MPI_Status status;
    int32_t length_message[1];
    
    // Receive the Total Length first
    MPI_Recv(length_message, 1, MPI_INT, source, GENOME_LENGTH_TAG, MPI_COMM_WORLD, &status);
    int32_t length = length_message[0];

    Log::info("receiving genome of length: %d from: %d\n", length, source);

    // Allocate memory for the full message
    char* genome_str = new char[length + 1];

    // Receive Data in 32KB Chunks
    // Loop until we have collected all 'length' bytes
    int32_t offset = 0;
    int32_t chunk_size = 32768;

    while (offset < length) {
        int32_t recv_size = length - offset;
        if (recv_size > chunk_size) {
            recv_size = chunk_size;
        }

        // Receive directly into the correct position in the buffer
        MPI_Recv(genome_str + offset, recv_size, MPI_CHAR, source, GENOME_TAG, MPI_COMM_WORLD, &status);
        offset += recv_size;
    }

    genome_str[length] = '\0';

    RNN_Genome* genome = new RNN_Genome(genome_str, length);

    delete[] genome_str;
    return genome;
}

void send_genome_to(int32_t target, RNN_Genome* genome) {
    char* byte_array;
    int32_t length;

    genome->write_to_array(&byte_array, length);

    Log::debug("sending genome of length: %d to: %d\n", length, target);

    // Send the Total Length
    int32_t length_message[1];
    length_message[0] = length;
    MPI_Send(length_message, 1, MPI_INT, target, GENOME_LENGTH_TAG, MPI_COMM_WORLD);

    // Send Data in 32KB Chunks
    // This bypasses the cluster's message size limit
    int32_t offset = 0;
    int32_t chunk_size = 32768; // 32KB chunk size is safe for all MPIs

    while (offset < length) {
        int32_t send_size = length - offset;
        if (send_size > chunk_size) {
            send_size = chunk_size;
        }
        
        // Send the specific chunk
        MPI_Send(byte_array + offset, send_size, MPI_CHAR, target, GENOME_TAG, MPI_COMM_WORLD);
        offset += send_size;
    }

    free(byte_array);
}

void send_terminate_message(int32_t target) {
    int32_t terminate_message[1];
    terminate_message[0] = 0;
    MPI_Send(terminate_message, 1, MPI_INT, target, TERMINATE_TAG, MPI_COMM_WORLD);
}

void receive_terminate_message(int32_t source) {
    MPI_Status status;
    int32_t terminate_message[1];
    MPI_Recv(terminate_message, 1, MPI_INT, source, TERMINATE_TAG, MPI_COMM_WORLD, &status);
}

#if 0
void master(int32_t max_rank) {
    // the "main" id will have already been set by the main function so we do not need to re-set it here
    Log::debug("MAX int32_t: %d\n", numeric_limits<int32_t>::max());

    int32_t terminates_sent = 0;

    while (true) {
        // wait for a incoming message
        MPI_Status status;
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        int32_t source = status.MPI_SOURCE;
        int32_t tag = status.MPI_TAG;
        Log::debug("probe returned message from: %d with tag: %d\n", source, tag);

        // if the message is a work request, send a genome

        if (tag == WORK_REQUEST_TAG) {
            receive_work_request(source);

            // if (transfer_learning_version.compare("v3") == 0 || transfer_learning_version.compare("v1+v3") == 0) {
            //     seed_stirs = 3;
            // }
            examm_mutex.lock();
            RNN_Genome* genome = examm->generate_genome();
            // --- NEW SWEET LOGIC: Add a COPY to the island's pool ---
            // We must copy it because the original 'genome' is deleted at the end of this block
            if (genome != NULL) {
                int32_t island_id = genome->get_group_id();
                examm->add_evaluating_genome(genome->copy());
            }
            // --------------------------------------------------------
            examm_mutex.unlock();

            if (genome == NULL) {  // search was completed if it returns NULL for an individual
                // send terminate message
                Log::info("terminating worker: %d\n", source);
                send_terminate_message(source);
                terminates_sent++;

                Log::debug("sent: %d terminates of %d\n", terminates_sent, (max_rank - 1));
                if (terminates_sent >= max_rank - 1) {
                    return;
                }

            } else {
                // genome->write_to_file( examm->get_output_directory() + "/before_send_gen_" +
                // to_string(genome->get_generation_id()) );

                // send genome
                Log::debug("sending genome to: %d\n", source);
                send_genome_to(source, genome);

                // delete this genome as it will not be used again
                delete genome;
            }
        } else if (tag == GENOME_LENGTH_TAG) {
            Log::debug("received genome from: %d\n", source);
            RNN_Genome* genome = receive_genome_from(source);

            examm_mutex.lock();
            
            // --- NEW SWEET LOGIC: Remove from the island's pool ---
            // Remove it BEFORE we insert it into the evaluated population
            int32_t island_id = genome->get_group_id();
            examm->remove_evaluating_genome(genome->copy());
            // ------------------------------------------------------

            examm->insert_genome(genome);
            examm_mutex.unlock();

            // delete the genome as it won't be used again, a copy was inserted
            delete genome;
            // this genome will be deleted if/when removed from population
        } else {
            Log::fatal("ERROR: received message from %d with unknown tag: %d", source, tag);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
}

void worker(int32_t rank) {
    Log::set_id("worker_" + to_string(rank));

    while (true) {
        Log::debug("sending work request!\n");
        send_work_request(0);
        Log::debug("sent work request!\n");

        MPI_Status status;
        MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        int32_t tag = status.MPI_TAG;

        Log::debug("probe received message with tag: %d\n", tag);

        if (tag == TERMINATE_TAG) {
            Log::debug("received terminate tag!\n");
            receive_terminate_message(0);
            break;

        } else if (tag == GENOME_LENGTH_TAG) {
            Log::debug("received genome!\n");
            RNN_Genome* genome = receive_genome_from(0);

            // have each worker write the backproagation to a separate log file
            string log_id = "genome_" + to_string(genome->get_generation_id()) + "_worker_" + to_string(rank);
            Log::set_id(log_id);
            genome->backpropagate_stochastic(
                training_inputs, training_outputs, validation_inputs, validation_outputs, weight_update_method
            );
            Log::release_id(log_id);

            // go back to the worker's log for MPI communication
            Log::set_id("worker_" + to_string(rank));

            send_genome_to(0, genome);

            delete genome;
        } else {
            Log::fatal("ERROR: received message with unknown tag: %d\n", tag);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // release the log file for the worker communication
    Log::release_id("worker_" + to_string(rank));
}
#endif

// =============================================================================
// FaultToleranceLogger
// Writes two files into the rank's output directory:
//   fault_tolerance_events.csv   — one row per event, every detail captured
//   fault_tolerance_summary.csv  — single-row aggregate written at run end
//
// Columns (events):
//   wall_time_s, event_type, reporting_rank, subject_rank,
//   active_ranks_count, best_fitness, genomes_evaluated,
//   downtime_s, detail
// =============================================================================
struct FaultToleranceLogger {
    std::ofstream events_file;
    std::ofstream summary_file;

    int32_t rank;
    std::chrono::steady_clock::time_point start_time;

    // Aggregates for summary.
    int32_t total_dropout_events    = 0;
    int32_t total_shutdown_events   = 0;
    int32_t total_recovery_events   = 0;
    int32_t total_peer_failures     = 0;
    int32_t peak_concurrent_failed  = 0;
    int32_t current_failed_count    = 0;
    double  total_downtime_s        = 0.0;
    double  fitness_at_first_failure = -1.0;
    double  token_recoveries        = 0;

    FaultToleranceLogger() = default;

    void open(const std::string& output_dir, int32_t r,
              std::chrono::steady_clock::time_point t0) {
        rank       = r;
        start_time = t0;

        events_file.open(output_dir + "/fault_tolerance_events.csv");
        events_file << "wall_time_s,event_type,reporting_rank,subject_rank,"
                    << "active_ranks_count,best_fitness,genomes_evaluated,"
                    << "downtime_s,detail\n";

        summary_file.open(output_dir + "/fault_tolerance_summary.csv");
    }

    void log(const std::string& event_type,
             int32_t subject_rank,
             size_t  active_count,
             double  best_fitness,
             int32_t genomes_evaluated,
             double  downtime_s  = -1.0,
             const std::string& detail = "")
    {
        if (!events_file.is_open()) return;

        const double wall_time =
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start_time).count();

        events_file << std::fixed << std::setprecision(4)
                    << wall_time          << ","
                    << event_type         << ","
                    << rank               << ","
                    << subject_rank       << ","
                    << active_count       << ","
                    << best_fitness       << ","
                    << genomes_evaluated  << ","
                    << downtime_s         << ","
                    << "\"" << detail << "\"\n";
        events_file.flush();
    }

    void write_summary(double total_wall_time_s,
                       double final_best_fitness,
                       int32_t total_genomes_evaluated)
    {
        if (!summary_file.is_open()) return;

        summary_file << "rank,total_wall_time_s,final_best_fitness,"
                     << "total_genomes_evaluated,"
                     << "total_dropout_events,total_shutdown_events,"
                     << "total_recovery_events,total_peer_failures_detected,"
                     << "peak_concurrent_failed,total_downtime_s,"
                     << "fitness_at_first_failure,token_recoveries\n";

        summary_file << std::fixed << std::setprecision(4)
                     << rank                    << ","
                     << total_wall_time_s        << ","
                     << final_best_fitness       << ","
                     << total_genomes_evaluated  << ","
                     << total_dropout_events     << ","
                     << total_shutdown_events    << ","
                     << total_recovery_events    << ","
                     << total_peer_failures      << ","
                     << peak_concurrent_failed   << ","
                     << total_downtime_s         << ","
                     << fitness_at_first_failure << ","
                     << token_recoveries         << "\n";
        summary_file.flush();
    }

    void on_failure(double best_fitness) {
        current_failed_count++;
        total_peer_failures++;
        if (current_failed_count > peak_concurrent_failed)
            peak_concurrent_failed = current_failed_count;
        if (fitness_at_first_failure < 0.0)
            fitness_at_first_failure = best_fitness;
    }

    void on_recovery(double downtime_s) {
        current_failed_count = std::max(0, current_failed_count - 1);
        total_recovery_events++;
        total_downtime_s += downtime_s;
    }
};

void peer_node(int32_t rank, int32_t max_rank, const DropoutConfig& dropout_cfg) {
    // Physical ring (fixed): used for heartbeats and token forwarding.
    // These never skip dropped ranks so the token always completes a full circuit.
    const int32_t ring_successor   = (rank + 1) % max_rank;
    const int32_t ring_predecessor = (rank - 1 + max_rank) % max_rank;

    // Active-ranks: sorted list of peers still participating in evolution.
    // Updated on PEER_FAILED, GRACEFUL_SHUTDOWN, and REJOIN_NOTIFY.
    // Controls MIGRATE ownership and BACKUP routing only.
    std::vector<int32_t> active_ranks(max_rank);
    std::iota(active_ranks.begin(), active_ranks.end(), 0);

    bool local_done        = false;
    bool consensus_reached = false;

    std::vector<IncomingGenomeTransfer> pending_incoming;
    std::vector<OutgoingGenomeTransfer> pending_outgoing;
    constexpr size_t MAX_PENDING_TRANSFERS = 8;

    const auto start_time = std::chrono::steady_clock::now();

    // Open fault-tolerance log files in this rank's output directory.
    FaultToleranceLogger ft_log;
    ft_log.open(examm->get_output_directory(), rank, start_time);

    // ----- Heartbeat state -----
    // Each rank sends a heartbeat to ring_successor every heartbeat_interval_ms.
    // Each rank monitors ring_predecessor — silence past heartbeat_timeout_ms = failure.
    auto last_heartbeat_sent     = start_time;
    auto last_heartbeat_from_pred = start_time;
    bool predecessor_declared_dead = false;
    // Don't fire the timeout until one full timeout window has elapsed
    // (gives all ranks time to start sending heartbeats on startup).
    const auto hb_warmup = std::chrono::milliseconds(dropout_cfg.heartbeat_timeout_ms);
    const auto hb_interval = std::chrono::milliseconds(dropout_cfg.heartbeat_interval_ms);
    const auto hb_timeout  = std::chrono::milliseconds(dropout_cfg.heartbeat_timeout_ms);

    // ----- Unintentional dropout state -----
    bool   is_dropped       = false;
    bool   dropout_sent     = false;
    bool   recovery_sent    = false;
    double dropout_time_s   = -1.0;

    // ----- Graceful shutdown state -----
    bool shutdown_sent = false;
    bool is_shutdown   = false;

    // ----- Token-ring recovery -----
    auto token_last_seen   = start_time;
    bool token_ever_seen   = false;
    const auto TOKEN_TIMEOUT = std::chrono::milliseconds(dropout_cfg.heartbeat_timeout_ms * 2);

    // Token fields: [0]=origin_rank [1]=hop_count [2]=done_count [3]=final_flag
    if (rank == 0 && max_rank > 1) {
        int32_t token[4] = {0, 0, 0, 0};
        MPI_Send(token, 4, MPI_INT, ring_successor, TERMINATION_TOKEN_TAG, MPI_COMM_WORLD);
        token_last_seen = std::chrono::steady_clock::now();
        token_ever_seen = true;
    }

    std::string peer_log_id = "peer_" + to_string(rank);
    Log::set_id(peer_log_id);

    while (!consensus_reached) {
        const auto   now       = std::chrono::steady_clock::now();
        const double elapsed_s = std::chrono::duration<double>(now - start_time).count();

        // =================================================================
        // UNINTENTIONAL DROPOUT TRIGGER
        // Rank goes silent — stops sending heartbeats.
        // Peers detect silence via timeout and broadcast PEER_FAILED.
        // No notification sent here; that's intentionally unrealistic.
        // =================================================================
        if (!dropout_sent
            && dropout_cfg.dropout_rank == rank
            && dropout_cfg.dropout_after_seconds > 0.0
            && elapsed_s >= dropout_cfg.dropout_after_seconds
            && !local_done)
        {
            // Flush best genome to successor before going dark (last known-good state).
            const int32_t backup_succ = next_alive_rank(rank, active_ranks);
            RNN_Genome*   best        = examm->get_best_genome();
            if (best != nullptr && backup_succ != rank
                && pending_outgoing.size() < MAX_PENDING_TRANSFERS)
            {
                queue_genome_send(GenomeTransferKind::BACKUP, backup_succ, best, pending_outgoing);
                while (!pending_outgoing.empty()) {
                    progress_outgoing(pending_outgoing);
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            }

            is_dropped     = true;
            dropout_sent   = true;
            local_done     = true;
            dropout_time_s = elapsed_s;

            // Remove self from active_ranks so our own routing stays consistent.
            active_ranks.erase(
                std::remove(active_ranks.begin(), active_ranks.end(), rank),
                active_ranks.end()
            );

            ft_log.log("DROPOUT_START", rank, active_ranks.size(),
                       examm->get_best_fitness(), (examm->get_best_genome() != nullptr ? examm->get_best_genome()->get_generation_id() : 0),
                       -1.0, "going_silent;peers_detect_via_heartbeat");
            ft_log.total_dropout_events++;
            if (ft_log.fitness_at_first_failure < 0.0)
                ft_log.fitness_at_first_failure = examm->get_best_fitness();

            Log::info("[DROPOUT] t=%.1fs  rank %d going silent (unintentional). "
                      "Peers will detect via heartbeat timeout.\n", elapsed_s, rank);
        }

        // =================================================================
        // GRACEFUL SHUTDOWN TRIGGER
        // Rank knows it is leaving permanently — broadcasts full state first.
        // =================================================================
        if (!shutdown_sent
            && dropout_cfg.shutdown_rank == rank
            && dropout_cfg.shutdown_after_seconds > 0.0
            && elapsed_s >= dropout_cfg.shutdown_after_seconds
            && !local_done)
        {
            Log::info("[SHUTDOWN] t=%.1fs  rank %d initiating graceful shutdown.\n", elapsed_s, rank);

            // Send best genome to ALL active peers so no work is lost.
            RNN_Genome* best = examm->get_best_genome();
            if (best != nullptr) {
                for (int32_t r : active_ranks) {
                    if (r != rank && pending_outgoing.size() < MAX_PENDING_TRANSFERS) {
                        Log::info("[SHUTDOWN] rank %d sending best genome (fitness=%.6f) to rank %d\n",
                                  rank, best->get_fitness(), r);
                        queue_genome_send(GenomeTransferKind::BACKUP, r, best, pending_outgoing);
                    }
                }
                while (!pending_outgoing.empty()) {
                    progress_outgoing(pending_outgoing);
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            }

            // Announce permanent departure to all active peers.
            for (int32_t r : active_ranks) {
                if (r != rank) {
                    MPI_Send(&rank, 1, MPI_INT, r, GRACEFUL_SHUTDOWN_TAG, MPI_COMM_WORLD);
                }
            }

            shutdown_sent = true;
            is_shutdown   = true;
            local_done    = true;

            active_ranks.erase(
                std::remove(active_ranks.begin(), active_ranks.end(), rank),
                active_ranks.end()
            );

            ft_log.log("GRACEFUL_SHUTDOWN_SENT", rank, active_ranks.size(),
                       examm->get_best_fitness(), (examm->get_best_genome() != nullptr ? examm->get_best_genome()->get_generation_id() : 0),
                       -1.0, "broadcast_best_genome_to_all_peers;permanent");
            ft_log.total_shutdown_events++;
            if (ft_log.fitness_at_first_failure < 0.0)
                ft_log.fitness_at_first_failure = examm->get_best_fitness();

            Log::info("[SHUTDOWN] rank %d permanently offline. %zu peer(s) remaining.\n",
                      rank, active_ranks.size());
        }

        // =================================================================
        // RECOVERY TRIGGER
        // After recovery_after_seconds past the dropout, this rank wakes up,
        // resumes heartbeats, and requests genome seed from the active ring.
        // =================================================================
        if (is_dropped && !recovery_sent
            && dropout_cfg.recovery_after_seconds > 0.0
            && elapsed_s >= dropout_time_s + dropout_cfg.recovery_after_seconds)
        {
            Log::info("[REJOIN] t=%.1fs  rank %d waking up and requesting rejoin.\n", elapsed_s, rank);

            is_dropped  = false;   // resume heartbeats
            local_done  = false;   // will be re-enabled once REJOIN_NOTIFY arrives

            // Request genome seed from the lowest active rank.
            if (!active_ranks.empty()) {
                int32_t target = active_ranks[0];
                MPI_Send(&rank, 1, MPI_INT, target, REJOIN_REQUEST_TAG, MPI_COMM_WORLD);
                Log::info("[REJOIN] rank %d sent REJOIN_REQUEST to rank %d\n", rank, target);
            }

            recovery_sent = true;
            last_heartbeat_from_pred = now;

            ft_log.log("REJOIN_REQUESTED", rank, active_ranks.size(),
                       examm->get_best_fitness(), (examm->get_best_genome() != nullptr ? examm->get_best_genome()->get_generation_id() : 0),
                       elapsed_s - dropout_time_s,
                       "downtime_s=" + std::to_string(elapsed_s - dropout_time_s));
        }

        // =================================================================
        // HEARTBEAT SEND
        // Skipped while this rank is dropped (silent) or permanently shut down.
        // =================================================================
        if (!is_dropped && !is_shutdown && max_rank > 1) {
            if (now - last_heartbeat_sent >= hb_interval) {
                MPI_Send(&rank, 1, MPI_INT, ring_successor, HEARTBEAT_TAG, MPI_COMM_WORLD);
                last_heartbeat_sent = now;
            }
        }

        // Progress background transfers.
        progress_outgoing(pending_outgoing);
        progress_incoming(pending_incoming, examm, rank, active_ranks);

        // =================================================================
        // HEARTBEAT RECEIVE
        // Update the timestamp of the last heartbeat from our predecessor.
        // =================================================================
        if (max_rank > 1) {
            int        flag_hb = 0;
            MPI_Status st_hb;
            MPI_Iprobe(ring_predecessor, HEARTBEAT_TAG, MPI_COMM_WORLD, &flag_hb, &st_hb);
            if (flag_hb) {
                int32_t hb_msg;
                MPI_Recv(&hb_msg, 1, MPI_INT, ring_predecessor,
                         HEARTBEAT_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                bool pred_active = std::find(active_ranks.begin(), active_ranks.end(),
                                             ring_predecessor) != active_ranks.end();
                if (pred_active) {
                    last_heartbeat_from_pred  = now;
                    predecessor_declared_dead = false;
                }
            }
        }

        // =================================================================
        // FAILURE DETECTION
        // If predecessor heartbeat has been silent past the timeout window,
        // this rank declares it failed and broadcasts PEER_FAILED to all.
        // =================================================================
        if (max_rank > 1 && !predecessor_declared_dead
            && now - start_time > hb_warmup)
        {
            bool pred_active = std::find(active_ranks.begin(), active_ranks.end(),
                                         ring_predecessor) != active_ranks.end();
            if (pred_active && now - last_heartbeat_from_pred > hb_timeout) {
                predecessor_declared_dead = true;

                const double silence_s =
                    std::chrono::duration<double>(now - last_heartbeat_from_pred).count();

                Log::info("[HEARTBEAT] rank %d: predecessor %d timed out (%.1fs silent). "
                          "Broadcasting PEER_FAILED.\n", rank, ring_predecessor, silence_s);

                // Broadcast to all other active ranks (not predecessor, not self).
                for (int32_t r : active_ranks) {
                    if (r != rank && r != ring_predecessor) {
                        MPI_Send(&ring_predecessor, 1, MPI_INT, r,
                                 PEER_FAILED_TAG, MPI_COMM_WORLD);
                    }
                }

                // Apply locally.
                active_ranks.erase(
                    std::remove(active_ranks.begin(), active_ranks.end(), ring_predecessor),
                    active_ranks.end()
                );

                ft_log.log("HEARTBEAT_TIMEOUT", ring_predecessor, active_ranks.size(),
                           examm->get_best_fitness(), (examm->get_best_genome() != nullptr ? examm->get_best_genome()->get_generation_id() : 0),
                           -1.0,
                           "silence_s=" + std::to_string(silence_s)
                           + ";broadcasting_peer_failed");
                ft_log.on_failure(examm->get_best_fitness());

                Log::info("[HEARTBEAT] rank %d: %zu active peer(s) after failure.\n",
                          rank, active_ranks.size());
            }
        }

        // =================================================================
        // PEER_FAILED RECEIVE
        // Another rank detected a failure and is broadcasting it.
        // =================================================================
        {
            int        flag_pf = 0;
            MPI_Status st_pf;
            MPI_Iprobe(MPI_ANY_SOURCE, PEER_FAILED_TAG, MPI_COMM_WORLD, &flag_pf, &st_pf);
            if (flag_pf) {
                int32_t failed_rank;
                MPI_Recv(&failed_rank, 1, MPI_INT, st_pf.MPI_SOURCE,
                         PEER_FAILED_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                bool still_active = std::find(active_ranks.begin(), active_ranks.end(),
                                              failed_rank) != active_ranks.end();
                if (still_active) {
                    Log::info("[PEER_FAILED] rank %d: rank %d declared failed by rank %d. "
                              "best_fitness=%.6f  active_before=%zu\n",
                              rank, failed_rank, st_pf.MPI_SOURCE,
                              examm->get_best_fitness(), active_ranks.size());

                    active_ranks.erase(
                        std::remove(active_ranks.begin(), active_ranks.end(), failed_rank),
                        active_ranks.end()
                    );

                    if (failed_rank == ring_predecessor) {
                        predecessor_declared_dead = true;
                    }

                    ft_log.log("PEER_FAILED_RECEIVED", failed_rank, active_ranks.size(),
                               examm->get_best_fitness(), (examm->get_best_genome() != nullptr ? examm->get_best_genome()->get_generation_id() : 0),
                               -1.0,
                               "detected_by_rank=" + std::to_string(st_pf.MPI_SOURCE));
                    ft_log.on_failure(examm->get_best_fitness());
                }
            }
        }

        // =================================================================
        // GRACEFUL_SHUTDOWN RECEIVE
        // A peer announced permanent departure with genome handoff.
        // =================================================================
        {
            int        flag_gs = 0;
            MPI_Status st_gs;
            MPI_Iprobe(MPI_ANY_SOURCE, GRACEFUL_SHUTDOWN_TAG, MPI_COMM_WORLD, &flag_gs, &st_gs);
            if (flag_gs) {
                int32_t shutting_rank;
                MPI_Recv(&shutting_rank, 1, MPI_INT, st_gs.MPI_SOURCE,
                         GRACEFUL_SHUTDOWN_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                Log::info("[SHUTDOWN] rank %d: rank %d permanently leaving. "
                          "best_fitness=%.6f  active_before=%zu\n",
                          rank, shutting_rank, examm->get_best_fitness(), active_ranks.size());

                active_ranks.erase(
                    std::remove(active_ranks.begin(), active_ranks.end(), shutting_rank),
                    active_ranks.end()
                );

                if (shutting_rank == ring_predecessor) {
                    predecessor_declared_dead = true;
                }

                ft_log.log("GRACEFUL_SHUTDOWN_RECEIVED", shutting_rank, active_ranks.size(),
                           examm->get_best_fitness(), (examm->get_best_genome() != nullptr ? examm->get_best_genome()->get_generation_id() : 0),
                           -1.0, "permanent;genome_handoff_received");
                ft_log.on_failure(examm->get_best_fitness());

                Log::info("[SHUTDOWN] rank %d: %zu active peer(s) remaining.\n",
                          rank, active_ranks.size());
            }
        }

        // =================================================================
        // REJOIN_REQUEST RECEIVE
        // A recovering rank is asking to re-enter the ring.
        // Respond with best genome and broadcast REJOIN_NOTIFY to all.
        // =================================================================
        {
            int        flag_rr = 0;
            MPI_Status st_rr;
            MPI_Iprobe(MPI_ANY_SOURCE, REJOIN_REQUEST_TAG, MPI_COMM_WORLD, &flag_rr, &st_rr);
            if (flag_rr) {
                int32_t rejoining_rank;
                MPI_Recv(&rejoining_rank, 1, MPI_INT, st_rr.MPI_SOURCE,
                         REJOIN_REQUEST_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                Log::info("[REJOIN] rank %d: rank %d requesting rejoin. Seeding genome.\n",
                          rank, rejoining_rank);

                RNN_Genome* best = examm->get_best_genome();
                if (best != nullptr && pending_outgoing.size() < MAX_PENDING_TRANSFERS) {
                    queue_genome_send(GenomeTransferKind::BACKUP, rejoining_rank, best, pending_outgoing);
                }

                if (std::find(active_ranks.begin(), active_ranks.end(), rejoining_rank)
                        == active_ranks.end()) {
                    active_ranks.push_back(rejoining_rank);
                    std::sort(active_ranks.begin(), active_ranks.end());
                }

                if (rejoining_rank == ring_predecessor) {
                    predecessor_declared_dead = false;
                    last_heartbeat_from_pred  = now;
                }

                for (int32_t r : active_ranks) {
                    if (r != rank) {
                        MPI_Send(&rejoining_rank, 1, MPI_INT, r,
                                 REJOIN_NOTIFY_TAG, MPI_COMM_WORLD);
                    }
                }

                ft_log.log("REJOIN_SEEDED", rejoining_rank, active_ranks.size(),
                           examm->get_best_fitness(), (examm->get_best_genome() != nullptr ? examm->get_best_genome()->get_generation_id() : 0),
                           -1.0, "sent_best_genome;broadcast_rejoin_notify");

                Log::info("[REJOIN] rank %d: rank %d reintegrated. active_ranks=%zu\n",
                          rank, rejoining_rank, active_ranks.size());
            }
        }

        // =================================================================
        // REJOIN_NOTIFY RECEIVE
        // A peer has been accepted back. Re-add to active_ranks.
        // If this is our own notification, resume evolution.
        // =================================================================
        {
            int        flag_rn = 0;
            MPI_Status st_rn;
            MPI_Iprobe(MPI_ANY_SOURCE, REJOIN_NOTIFY_TAG, MPI_COMM_WORLD, &flag_rn, &st_rn);
            if (flag_rn) {
                int32_t rejoining_rank;
                MPI_Recv(&rejoining_rank, 1, MPI_INT, st_rn.MPI_SOURCE,
                         REJOIN_NOTIFY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                if (std::find(active_ranks.begin(), active_ranks.end(), rejoining_rank)
                        == active_ranks.end()) {
                    active_ranks.push_back(rejoining_rank);
                    std::sort(active_ranks.begin(), active_ranks.end());
                }

                if (rejoining_rank == ring_predecessor) {
                    predecessor_declared_dead = false;
                    last_heartbeat_from_pred  = now;
                }

                if (rejoining_rank == rank) {
                    local_done = false;
                    const double downtime_s = elapsed_s - dropout_time_s;
                    ft_log.log("REJOIN_COMPLETE", rank, active_ranks.size(),
                               examm->get_best_fitness(), (examm->get_best_genome() != nullptr ? examm->get_best_genome()->get_generation_id() : 0),
                               downtime_s,
                               "downtime_s=" + std::to_string(downtime_s)
                               + ";resuming_evolution");
                    ft_log.on_recovery(downtime_s);
                    Log::info("[REJOIN] rank %d: rejoin confirmed. Resuming evolution. "
                              "active_ranks=%zu\n", rank, active_ranks.size());
                } else {
                    ft_log.log("REJOIN_NOTIFY_RECEIVED", rejoining_rank, active_ranks.size(),
                               examm->get_best_fitness(), (examm->get_best_genome() != nullptr ? examm->get_best_genome()->get_generation_id() : 0));
                    Log::info("[REJOIN] rank %d: rank %d has rejoined. active_ranks=%zu\n",
                              rank, rejoining_rank, active_ranks.size());
                }
            }
        }

        // =================================================================
        // TOKEN-RING TERMINATION CONSENSUS (physical ring, unchanged logic).
        // =================================================================
        {
            int        flag_token = 0;
            MPI_Status st_token;
            MPI_Iprobe(MPI_ANY_SOURCE, TERMINATION_TOKEN_TAG, MPI_COMM_WORLD, &flag_token, &st_token);
            if (flag_token) {
                int32_t token[4];
                MPI_Recv(token, 4, MPI_INT, st_token.MPI_SOURCE,
                         TERMINATION_TOKEN_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                token_last_seen = std::chrono::steady_clock::now();
                token_ever_seen = true;

                const int32_t origin     = token[0];
                int32_t       hop        = token[1];
                int32_t       done       = token[2];
                int32_t       final_flag = token[3];

                if (final_flag != 0) {
                    consensus_reached = true;
                    if (ring_successor != rank) {
                        MPI_Send(token, 4, MPI_INT, ring_successor,
                                 TERMINATION_TOKEN_TAG, MPI_COMM_WORLD);
                    }
                    break;
                }

                if (local_done) done += 1;
                hop += 1;

                if (hop >= max_rank && rank == origin) {
                    token[2] = done;
                    if (done >= max_rank) {
                        token[3] = 1;
                    } else {
                        token[2] = 0;
                    }
                    token[1] = 0;
                } else {
                    token[1] = hop;
                    token[2] = done;
                }

                MPI_Send(token, 4, MPI_INT, ring_successor,
                         TERMINATION_TOKEN_TAG, MPI_COMM_WORLD);
            }
        }

        // =================================================================
        // TOKEN-RING RECOVERY
        // If the token is lost (holder dropped before forwarding), the lowest
        // active rank regenerates it after TOKEN_TIMEOUT of silence.
        // =================================================================
        if (token_ever_seen && max_rank > 1) {
            if (std::chrono::steady_clock::now() - token_last_seen > TOKEN_TIMEOUT) {
                const bool i_am_lowest = !active_ranks.empty() && active_ranks[0] == rank;
                if (i_am_lowest) {
                    const double silence_s = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - token_last_seen).count();
                    Log::info("[TOKEN] rank %d regenerating lost termination token "
                              "(%.1fs silent).\n", rank, silence_s);
                    int32_t token[4] = {rank, 0, local_done ? 1 : 0, 0};
                    MPI_Send(token, 4, MPI_INT, ring_successor,
                             TERMINATION_TOKEN_TAG, MPI_COMM_WORLD);
                    token_last_seen = std::chrono::steady_clock::now();

                    ft_log.log("TOKEN_RECOVERED", rank, active_ranks.size(),
                               examm->get_best_fitness(), (examm->get_best_genome() != nullptr ? examm->get_best_genome()->get_generation_id() : 0),
                               -1.0, "silence_s=" + std::to_string(silence_s));
                    ft_log.token_recoveries++;
                }
            }
        }

        // =================================================================
        // GENOME TRANSFER POLLING
        // =================================================================
        while (pending_incoming.size() < MAX_PENDING_TRANSFERS) {
            int        migrate_flag = 0;
            MPI_Status st_migrate;
            MPI_Iprobe(MPI_ANY_SOURCE, MIGRATE_GENOME_TAG, MPI_COMM_WORLD, &migrate_flag, &st_migrate);
            if (!migrate_flag) break;
            int32_t length = 0;
            MPI_Recv(&length, 1, MPI_INT, st_migrate.MPI_SOURCE,
                     MIGRATE_GENOME_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            post_genome_receive(GenomeTransferKind::MIGRATE, st_migrate.MPI_SOURCE,
                                length, pending_incoming);
        }

        while (pending_incoming.size() < MAX_PENDING_TRANSFERS) {
            int        backup_flag = 0;
            MPI_Status st_backup;
            MPI_Iprobe(MPI_ANY_SOURCE, BACKUP_GENOME_TAG, MPI_COMM_WORLD, &backup_flag, &st_backup);
            if (!backup_flag) break;
            int32_t length = 0;
            MPI_Recv(&length, 1, MPI_INT, st_backup.MPI_SOURCE,
                     BACKUP_GENOME_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            post_genome_receive(GenomeTransferKind::BACKUP, st_backup.MPI_SOURCE,
                                length, pending_incoming);
        }

        if (max_rank == 1 && local_done) {
            consensus_reached = true;
            break;
        }

        // =================================================================
        // GENOME EVALUATION
        // Skipped while this rank is dropped, shut down, or awaiting rejoin.
        // =================================================================
        if (!local_done && pending_incoming.empty()) {
            RNN_Genome* genome = examm->generate_genome();
            if (genome == NULL) {
                local_done = true;
                continue;
            }

            examm->add_evaluating_genome(genome);

            const std::string eval_log_id =
                "peer_eval_" + to_string(genome->get_generation_id()) + "_rank_" + to_string(rank);
            Log::set_id(eval_log_id);
            genome->backpropagate_stochastic(
                training_inputs, training_outputs,
                validation_inputs, validation_outputs,
                weight_update_method
            );
            Log::release_id(eval_log_id);

            examm->remove_evaluating_genome(genome);

            const double prev_best          = examm->get_best_fitness();
            const bool   inserted           = examm->insert_genome(genome);
            const bool   is_new_global_best = inserted && genome->get_fitness() < prev_best;

            if (is_new_global_best) {
                const int32_t backup_succ = next_alive_rank(rank, active_ranks);
                if (backup_succ != rank && pending_outgoing.size() < MAX_PENDING_TRANSFERS) {
                    queue_genome_send(GenomeTransferKind::BACKUP, backup_succ, genome, pending_outgoing);
                }

                const int32_t owner = genome_owner_rank_dynamic(genome, active_ranks);
                if (owner != rank && pending_outgoing.size() < MAX_PENDING_TRANSFERS) {
                    queue_genome_send(GenomeTransferKind::MIGRATE, owner, genome, pending_outgoing);
                }
            }

            delete genome;
            Log::set_id(peer_log_id);
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    // Drain any in-flight transfers before exit.
    for (auto& t : pending_incoming) {
        if (!t.requests.empty()) {
            MPI_Waitall((int) t.requests.size(), t.requests.data(), MPI_STATUSES_IGNORE);
        }
    }
    progress_incoming(pending_incoming, examm, rank, active_ranks);

    for (auto& t : pending_outgoing) {
        if (!t.requests.empty()) {
            MPI_Waitall((int) t.requests.size(), t.requests.data(), MPI_STATUSES_IGNORE);
        }
    }
    progress_outgoing(pending_outgoing);

    // Write summary row now that the run is complete.
    const double total_wall_time =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start_time).count();
    ft_log.write_summary(total_wall_time,
                         examm->get_best_fitness(),
                         (examm->get_best_genome() != nullptr ? examm->get_best_genome()->get_generation_id() : 0));

    Log::release_id(peer_log_id);
}

#ifndef EXAMM_MPI_P2P_UNIT_TEST
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int32_t rank = 0, max_rank = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &max_rank);

    arguments = vector<string>(argv, argv + argc);

    Log::initialize(arguments);
    Log::set_rank(rank);
    Log::set_id("main_" + to_string(rank));
    Log::restrict_to_rank(0);

    TimeSeriesSets* time_series_sets = TimeSeriesSets::generate_from_arguments(arguments);
    get_train_validation_data(
        arguments,
        time_series_sets,
        training_inputs,
        training_outputs,
        validation_inputs,
        validation_outputs
    );

    weight_update_method = new WeightUpdate();
    weight_update_method->generate_from_arguments(arguments);

    WeightRules* weight_rules = new WeightRules();
    weight_rules->initialize_from_args(arguments);

    // Seed genome must be identical on every rank; generate it only on rank 0
    // and broadcast the bytes to all peers.
    RNN_Genome* seed_genome = nullptr;
    if (rank == 0) {
        seed_genome = get_seed_genome(arguments, time_series_sets, weight_rules);
    }
    seed_genome = broadcast_genome_seed(seed_genome, rank, max_rank);
    if (seed_genome == nullptr) {
        Log::fatal("Failed to broadcast seed genome to rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Ensure each peer writes EXAMM artifacts to its own output directory.
    std::string base_output_directory = "";
    for (size_t i = 0; i + 1 < arguments.size(); i++) {
        if (arguments[i].compare("--output_directory") == 0) {
            base_output_directory = arguments[i + 1];
            break;
        }
    }
    if (base_output_directory.size() == 0) {
        base_output_directory = "./output";
    }
    std::string peer_output_directory = base_output_directory + "/p2p_rank_" + to_string(rank);
    std::vector<std::string> peer_arguments = with_peer_output_directory(arguments, peer_output_directory);

    if (rank == 0) {
        write_time_series_to_file(arguments, time_series_sets);
    }

    Log::clear_rank_restriction();

    // Fault-tolerance simulation flags (all optional).
    //
    // Unintentional failure (heartbeat-detected, node recovers):
    //   --dropout_rank <r>               rank that goes silent
    //   --dropout_after_seconds <t>      when it goes silent
    //   --recovery_after_seconds <t>     seconds after dropout before it rejoins
    //
    // Intentional/graceful shutdown (node broadcasts state then leaves permanently):
    //   --shutdown_rank <r>              rank that shuts down
    //   --shutdown_after_seconds <t>     when it shuts down
    //
    // Heartbeat tuning:
    //   --heartbeat_interval_ms <ms>     ping frequency (default 1000)
    //   --heartbeat_timeout_ms  <ms>     silence before failure declared (default 5000)
    DropoutConfig dropout_cfg;
    get_argument(arguments, "--dropout_rank",           false, dropout_cfg.dropout_rank);
    get_argument(arguments, "--dropout_after_seconds",  false, dropout_cfg.dropout_after_seconds);
    get_argument(arguments, "--recovery_after_seconds", false, dropout_cfg.recovery_after_seconds);
    get_argument(arguments, "--shutdown_rank",          false, dropout_cfg.shutdown_rank);
    get_argument(arguments, "--shutdown_after_seconds", false, dropout_cfg.shutdown_after_seconds);
    get_argument(arguments, "--heartbeat_interval_ms",  false, dropout_cfg.heartbeat_interval_ms);
    get_argument(arguments, "--heartbeat_timeout_ms",   false, dropout_cfg.heartbeat_timeout_ms);

    if (rank == 0) {
        if (dropout_cfg.dropout_rank >= 0) {
            Log::info("[SIM] Unintentional failure: rank %d silent at t=%.1fs, "
                      "recovers after %.1fs\n",
                      dropout_cfg.dropout_rank, dropout_cfg.dropout_after_seconds,
                      dropout_cfg.recovery_after_seconds);
        }
        if (dropout_cfg.shutdown_rank >= 0) {
            Log::info("[SIM] Graceful shutdown: rank %d permanently leaves at t=%.1fs\n",
                      dropout_cfg.shutdown_rank, dropout_cfg.shutdown_after_seconds);
        }
    }

    examm = generate_examm_from_arguments(peer_arguments, time_series_sets, weight_rules, seed_genome);
    peer_node(rank, max_rank, dropout_cfg);

    Log::set_id("main_" + to_string(rank));
    finished = true;
    Log::debug("rank %d completed!\n");
    Log::release_id("main_" + to_string(rank));
    MPI_Finalize();

    delete time_series_sets;
    return 0;
}
#endif
