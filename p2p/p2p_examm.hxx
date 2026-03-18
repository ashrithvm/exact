#ifndef P2P_EXAMM_HXX
#define P2P_EXAMM_HXX

#include <atomic>
#include <chrono>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "examm/examm.hxx"
#include "mpi.h"
#include "p2p_island_strategy.hxx"
#include "p2p_tags.hxx"
#include "peer_ring.hxx"
#include "replica_store.hxx"
#include "rnn/rnn_genome.hxx"
#include "weights/weight_update.hxx"

using std::atomic;
using std::map;
using std::mutex;
using std::string;
using std::vector;

/**
 * Main P2P coordinator for a single peer.
 *
 * Each peer owns a subset of islands, runs a local EXAMM instance,
 * and communicates directly with other peers for inter-island crossover,
 * genome replication, and distributed coordination.
 *
 * Manages two threads:
 *   Thread 1 (evolution_loop): generates genomes, trains them, inserts them
 *   Thread 2 (message_handler): handles incoming MPI messages from other peers
 */
class P2PExamm {
   private:
    int32_t rank;
    int32_t num_peers;

    EXAMM* local_examm;
    PeerRing* ring;
    ReplicaStore* replica_store;
    P2PIslandSpeciationStrategy* p2p_strategy;

    mutex examm_mutex;
    atomic<bool> terminated;
    int32_t local_evaluated_count;
    int32_t max_genomes;

    int32_t is_sweet;

    // Training data references (all peers load all data)
    vector<vector<vector<double>>>& training_inputs;
    vector<vector<vector<double>>>& training_outputs;
    vector<vector<vector<double>>>& validation_inputs;
    vector<vector<vector<double>>>& validation_outputs;
    WeightUpdate* weight_update_method;

    // Heartbeat tracking
    map<int32_t, std::chrono::time_point<std::chrono::steady_clock>> last_heartbeat;
    std::chrono::time_point<std::chrono::steady_clock> last_heartbeat_sent;

   public:
    P2PExamm(
        int32_t _rank, int32_t _num_peers, EXAMM* _local_examm,
        P2PIslandSpeciationStrategy* _p2p_strategy, PeerRing* _ring,
        int32_t _max_genomes, int32_t _is_sweet,
        vector<vector<vector<double>>>& _training_inputs,
        vector<vector<vector<double>>>& _training_outputs,
        vector<vector<vector<double>>>& _validation_inputs,
        vector<vector<vector<double>>>& _validation_outputs,
        WeightUpdate* _weight_update_method
    );

    ~P2PExamm();

    /**
     * Launches evolution_loop and message_handler threads, blocks until done.
     */
    void run();

    /**
     * Thread 1: Evolution + Training loop.
     * Generates genomes, trains them via backpropagation, inserts results.
     */
    void evolution_loop();

    /**
     * Thread 2: Incoming MPI message handler loop.
     * Responds to crossover requests, replication, heartbeats, termination.
     */
    void message_handler();

    /**
     * Request a genome from a remote peer's island for inter-island crossover.
     * This is a blocking call: sends CROSSOVER_REQUEST, waits for response.
     * Must NOT be called while examm_mutex is held (would deadlock with handler).
     */
    RNN_Genome* request_remote_genome(int32_t global_island_id);

    /**
     * Handle an incoming crossover request from another peer.
     * Picks a random genome from the requested island and sends it back.
     */
    void handle_crossover_request(int32_t source);

    /**
     * Replicate a genome to k successor peers for fault tolerance.
     */
    void replicate_genome(RNN_Genome* genome, int32_t global_island_id);

    /**
     * Handle an incoming replicated genome from another peer.
     */
    void handle_replication(int32_t source);

    /**
     * Periodic distributed termination check via MPI_Allreduce.
     */
    void check_termination();

    /**
     * Periodic best genome fitness sharing via MPI_Allreduce.
     */
    void sync_best_genome();

    /**
     * Send heartbeats to predecessor and successor on the ring.
     */
    void send_heartbeats();

    /**
     * Handle an incoming heartbeat.
     */
    void handle_heartbeat(int32_t source);

    /**
     * Check if any neighbors have timed out and handle their failure.
     */
    void check_peer_liveness();

    /**
     * Handle the failure of a peer — take over its islands from replica store.
     */
    void handle_peer_failure(int32_t failed_peer);
};

#endif
