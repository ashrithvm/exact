#include <chrono>
#include <iomanip>
using std::fixed;
using std::setprecision;
using std::setw;

#include <cstdint>
#include <algorithm>

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
#define SEED_GENOME_LENGTH_TAG  8
#define SEED_GENOME_DATA_TAG     9

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
    int32_t max_rank
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
            int32_t owner = genome_owner_rank(genome, max_rank);
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

void peer_node(int32_t rank, int32_t max_rank) {
    const int32_t successor = (rank + 1) % max_rank;

    bool local_done = false;
    bool consensus_reached = false;

    std::vector<IncomingGenomeTransfer> pending_incoming;
    std::vector<OutgoingGenomeTransfer> pending_outgoing;
    constexpr size_t MAX_PENDING_TRANSFERS = 8;

    // Start the distributed token-ring from rank 0.
    // Token fields: [0]=origin_rank, [1]=hop_count, [2]=done_count, [3]=final_flag
    if (rank == 0 && max_rank > 1) {
        int32_t token[4] = {0, 0, 0, 0};
        MPI_Send(token, 4, MPI_INT, successor, TERMINATION_TOKEN_TAG, MPI_COMM_WORLD);
    }

    std::string peer_log_id = "peer_" + to_string(rank);
    Log::set_id(peer_log_id);

    while (!consensus_reached) {
        // Progress background transfers first.
        progress_outgoing(pending_outgoing);
        progress_incoming(pending_incoming, examm, rank, max_rank);

        // Poll and handle termination token.
        int flag_token = 0;
        MPI_Status st_token;
        MPI_Iprobe(MPI_ANY_SOURCE, TERMINATION_TOKEN_TAG, MPI_COMM_WORLD, &flag_token, &st_token);
        if (flag_token) {
            int32_t token[4];
            MPI_Recv(
                token, 4, MPI_INT, st_token.MPI_SOURCE, TERMINATION_TOKEN_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE
            );

            const int32_t origin = token[0];
            int32_t hop = token[1];
            int32_t done = token[2];
            int32_t final_flag = token[3];

            if (final_flag != 0) {
                consensus_reached = true;
                MPI_Send(token, 4, MPI_INT, successor, TERMINATION_TOKEN_TAG, MPI_COMM_WORLD);
                break;
            }

            if (local_done) {
                done += 1;
            }

            hop += 1;

            // End of cycle happens when token returns to origin after max_rank hops.
            if (hop >= max_rank && rank == origin) {
                token[2] = done;
                if (done >= max_rank) {
                    token[3] = 1;  // final
                } else {
                    // Next cycle: reset the counts.
                    token[2] = 0;
                }
                token[1] = 0;
            } else {
                token[1] = hop;
                token[2] = done;
            }

            MPI_Send(token, 4, MPI_INT, successor, TERMINATION_TOKEN_TAG, MPI_COMM_WORLD);
        }

        // Post receives for incoming genome payloads (length headers are small).
        // We only post when we have capacity for more outstanding transfers.
        while (pending_incoming.size() < MAX_PENDING_TRANSFERS) {
            int migrate_flag = 0;
            MPI_Status st_migrate;
            MPI_Iprobe(MPI_ANY_SOURCE, MIGRATE_GENOME_TAG, MPI_COMM_WORLD, &migrate_flag, &st_migrate);
            if (!migrate_flag) break;

            int32_t length = 0;
            MPI_Recv(&length, 1, MPI_INT, st_migrate.MPI_SOURCE, MIGRATE_GENOME_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            post_genome_receive(
                GenomeTransferKind::MIGRATE, st_migrate.MPI_SOURCE, length, pending_incoming
            );
        }

        while (pending_incoming.size() < MAX_PENDING_TRANSFERS) {
            int backup_flag = 0;
            MPI_Status st_backup;
            MPI_Iprobe(MPI_ANY_SOURCE, BACKUP_GENOME_TAG, MPI_COMM_WORLD, &backup_flag, &st_backup);
            if (!backup_flag) break;

            int32_t length = 0;
            MPI_Recv(&length, 1, MPI_INT, st_backup.MPI_SOURCE, BACKUP_GENOME_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            post_genome_receive(
                GenomeTransferKind::BACKUP, st_backup.MPI_SOURCE, length, pending_incoming
            );
        }

        // If there are no in-flight incoming genomes, evaluate the next local one.
        if (max_rank == 1 && local_done) {
            consensus_reached = true;
            break;
        }

        if (!local_done && pending_incoming.empty()) {
            RNN_Genome* genome = examm->generate_genome();
            if (genome == NULL) {
                local_done = true;
                continue;
            }

            examm->add_evaluating_genome(genome);

            const std::string eval_log_id = "peer_eval_" + to_string(genome->get_generation_id()) + "_rank_" + to_string(rank);
            Log::set_id(eval_log_id);
            genome->backpropagate_stochastic(
                training_inputs,
                training_outputs,
                validation_inputs,
                validation_outputs,
                weight_update_method
            );
            Log::release_id(eval_log_id);

            // Remove from SWEET evaluating pool *before* inserting into evaluated population.
            examm->remove_evaluating_genome(genome);

            // Capture previous best fitness before insert for migration/backup decisions.
            const double prev_best = examm->get_best_fitness();
            const bool inserted = examm->insert_genome(genome);

            // Ownership-based migration: only migrate the high-performing (new-global-best) genomes.
            const bool is_new_global_best = inserted && genome->get_fitness() < prev_best;

            if (is_new_global_best) {
                if (successor != rank && pending_outgoing.size() < MAX_PENDING_TRANSFERS) {
                    queue_genome_send(
                        GenomeTransferKind::BACKUP, successor, genome, pending_outgoing
                    );
                }

                const int32_t owner = genome_owner_rank(genome, max_rank);
                if (owner != rank && pending_outgoing.size() < MAX_PENDING_TRANSFERS) {
                    queue_genome_send(
                        GenomeTransferKind::MIGRATE, owner, genome, pending_outgoing
                    );
                }
            }

            delete genome;

            // Restore the peer log destination for subsequent MPI activity.
            Log::set_id(peer_log_id);
        } else {
            // Avoid a tight spin when waiting for messages/transfers.
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    // Drain any in-flight transfers so we can exit cleanly.
    for (auto& t : pending_incoming) {
        if (!t.requests.empty()) {
            MPI_Waitall((int) t.requests.size(), t.requests.data(), MPI_STATUSES_IGNORE);
        }
    }
    progress_incoming(pending_incoming, examm, rank, max_rank);

    for (auto& t : pending_outgoing) {
        if (!t.requests.empty()) {
            MPI_Waitall((int) t.requests.size(), t.requests.data(), MPI_STATUSES_IGNORE);
        }
    }
    progress_outgoing(pending_outgoing);

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

    examm = generate_examm_from_arguments(peer_arguments, time_series_sets, weight_rules, seed_genome);
    peer_node(rank, max_rank);

    Log::set_id("main_" + to_string(rank));
    finished = true;
    Log::debug("rank %d completed!\n");
    Log::release_id("main_" + to_string(rank));
    MPI_Finalize();

    delete time_series_sets;
    return 0;
}
#endif
