#include "p2p_examm.hxx"

#include <thread>
#include <unistd.h>

#include "common/log.hxx"
#include "p2p_comm.hxx"
#include "p2p_tags.hxx"

using std::thread;

P2PExamm::P2PExamm(
    int32_t _rank, int32_t _num_peers, EXAMM* _local_examm,
    P2PIslandSpeciationStrategy* _p2p_strategy, PeerRing* _ring,
    int32_t _max_genomes, int32_t _is_sweet,
    vector<vector<vector<double>>>& _training_inputs,
    vector<vector<vector<double>>>& _training_outputs,
    vector<vector<vector<double>>>& _validation_inputs,
    vector<vector<vector<double>>>& _validation_outputs,
    WeightUpdate* _weight_update_method
)
    : rank(_rank),
      num_peers(_num_peers),
      local_examm(_local_examm),
      ring(_ring),
      p2p_strategy(_p2p_strategy),
      terminated(false),
      local_evaluated_count(0),
      max_genomes(_max_genomes),
      is_sweet(_is_sweet),
      training_inputs(_training_inputs),
      training_outputs(_training_outputs),
      validation_inputs(_validation_inputs),
      validation_outputs(_validation_outputs),
      weight_update_method(_weight_update_method) {
    int32_t island_size = 0;
    vector<int32_t> owned = ring->get_owned_islands(rank);
    if (!owned.empty()) {
        island_size = p2p_strategy->get_island_by_index(0)->get_max_size();
    }
    replica_store = new ReplicaStore(island_size > 0 ? island_size : 10);

    // Initialize heartbeat tracking for neighbors
    auto now = std::chrono::steady_clock::now();
    last_heartbeat_sent = now;
    int32_t successor = ring->get_successor(rank);
    int32_t predecessor = ring->get_predecessor(rank);
    if (successor != rank) last_heartbeat[successor] = now;
    if (predecessor != rank && predecessor != successor) last_heartbeat[predecessor] = now;

    // Set up the remote genome requester callback for the P2P strategy
    // Note: this callback will be called from Thread 1 while examm_mutex is NOT held,
    // because the MPI communication in request_remote_genome is blocking
    p2p_strategy->set_remote_genome_requester(
        [this](int32_t global_island_id) -> RNN_Genome* {
            return this->request_remote_genome(global_island_id);
        }
    );
}

P2PExamm::~P2PExamm() {
    delete replica_store;
}

void P2PExamm::run() {
    Log::info("P2P peer %d starting with %d owned islands\n",
              rank, (int32_t)ring->get_owned_islands(rank).size());

    // Launch message handler in a separate thread
    thread handler_thread(&P2PExamm::message_handler, this);

    // Run evolution loop in the main thread
    evolution_loop();

    // Signal handler thread to stop and wait for it
    terminated = true;
    handler_thread.join();

    Log::info("P2P peer %d finished. Evaluated %d genomes locally.\n", rank, local_evaluated_count);
}

void P2PExamm::evolution_loop() {
    Log::set_id("peer_" + std::to_string(rank) + "_evolution");

    while (!terminated) {
        // Generate a genome
        // NOTE: We must NOT hold examm_mutex during generate_genome because the
        // P2P strategy's generate_for_filled_island may call request_remote_genome,
        // which does blocking MPI communication. If we held the mutex, the message
        // handler thread couldn't respond to incoming crossover requests, causing
        // deadlock when two peers request genomes from each other simultaneously.
        //
        // This is safe because:
        // 1. Only this thread calls generate_genome (single evolution thread)
        // 2. The message handler thread only reads island data for crossover responses
        //    (get_best_genome->copy), which is read-only on the population
        // 3. Insert operations that modify population state are still mutex-protected
        RNN_Genome* genome = NULL;
        {
            examm_mutex.lock();
            // generate_genome may call generate_for_filled_island which may need
            // to request a remote genome. We unlock before that happens.
            // The P2P strategy's remote_genome_requester callback does the MPI comm.
            examm_mutex.unlock();

            // Call generate_genome without holding the lock
            // The strategy's internal state (generation_island counter, etc.) is only
            // modified by this thread, so no race condition.
            genome = local_examm->generate_genome();
        }

        if (genome != NULL && is_sweet) {
            examm_mutex.lock();
            local_examm->add_evaluating_genome(genome->copy());
            examm_mutex.unlock();
        }

        if (genome == NULL) {
            // Local termination condition met — initiate distributed check
            Log::info("P2P peer %d: local generate_genome returned NULL, checking global termination\n", rank);
            check_termination();
            if (!terminated) {
                // Other peers may still have work; sleep briefly and retry
                usleep(100000);  // 100ms
            }
            continue;
        }

        // Train the genome (no lock needed — purely local computation)
        string log_id = "genome_" + std::to_string(genome->get_generation_id()) + "_peer_" + std::to_string(rank);
        Log::set_id(log_id);
        genome->backpropagate_stochastic(
            training_inputs, training_outputs, validation_inputs, validation_outputs, weight_update_method
        );
        Log::release_id(log_id);
        Log::set_id("peer_" + std::to_string(rank) + "_evolution");

        // Insert the trained genome (must hold lock — modifies population)
        examm_mutex.lock();
        if (is_sweet) {
            local_examm->remove_evaluating_genome(genome->copy());
        }
        local_examm->insert_genome(genome);
        local_evaluated_count++;
        examm_mutex.unlock();

        // Replicate to successors for fault tolerance (no lock needed)
        replicate_genome(genome, genome->get_group_id());

        // Periodic distributed coordination
        if (local_evaluated_count % SYNC_INTERVAL == 0) {
            check_termination();
            sync_best_genome();
        }

        delete genome;
    }

    Log::release_id("peer_" + std::to_string(rank) + "_evolution");
}

void P2PExamm::message_handler() {
    Log::set_id("peer_" + std::to_string(rank) + "_handler");

    while (!terminated) {
        MPI_Status status;
        int flag = 0;
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);

        if (!flag) {
            // No message available — do periodic maintenance
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_heartbeat_sent).count();
            if (elapsed >= HEARTBEAT_INTERVAL_S) {
                send_heartbeats();
                check_peer_liveness();
                last_heartbeat_sent = now;
            }
            usleep(1000);  // 1ms — avoid busy-wait
            continue;
        }

        int32_t source = status.MPI_SOURCE;
        int32_t tag = status.MPI_TAG;

        switch (tag) {
            case CROSSOVER_REQUEST_TAG:
                handle_crossover_request(source);
                break;

            case REPLICATION_ISLAND_ID_TAG:
                handle_replication(source);
                break;

            case HEARTBEAT_TAG:
                handle_heartbeat(source);
                break;

            case TERMINATION_CONFIRM_TAG:
                p2p_receive_termination_confirm(source);
                Log::info("P2P peer %d: received termination confirmation from peer %d\n", rank, source);
                terminated = true;
                break;

            default:
                Log::warning("P2P peer %d: received message with unknown tag %d from %d\n", rank, tag, source);
                // Consume the message to avoid blocking
                int count;
                MPI_Get_count(&status, MPI_CHAR, &count);
                if (count > 0) {
                    char* buf = new char[count];
                    MPI_Recv(buf, count, MPI_CHAR, source, tag, MPI_COMM_WORLD, &status);
                    delete[] buf;
                }
                break;
        }
    }

    Log::release_id("peer_" + std::to_string(rank) + "_handler");
}

RNN_Genome* P2PExamm::request_remote_genome(int32_t global_island_id) {
    int32_t owner = ring->get_owner(global_island_id);
    if (owner < 0 || !ring->is_active(owner)) {
        Log::warning("P2P peer %d: cannot request genome from island %d — owner %d is not active\n",
                     rank, global_island_id, owner);
        return NULL;
    }

    Log::debug("P2P peer %d: requesting genome from island %d on peer %d\n",
               rank, global_island_id, owner);

    // Send crossover request
    p2p_send_crossover_request(owner, global_island_id);

    // Wait for response — the remote peer will send back a genome
    RNN_Genome* genome = p2p_receive_genome(owner, CROSSOVER_RESPONSE_LENGTH_TAG, CROSSOVER_RESPONSE_TAG);

    Log::debug("P2P peer %d: received genome from island %d on peer %d\n",
               rank, global_island_id, owner);

    return genome;
}

void P2PExamm::handle_crossover_request(int32_t source) {
    int32_t requested_island = p2p_receive_crossover_request(source);

    Log::debug("P2P peer %d: handling crossover request for island %d from peer %d\n",
               rank, requested_island, source);

    // Find the local index for this global island ID
    vector<int32_t> global_ids = p2p_strategy->get_global_island_ids();
    int32_t local_index = -1;
    for (int32_t i = 0; i < (int32_t)global_ids.size(); i++) {
        if (global_ids[i] == requested_island) {
            local_index = i;
            break;
        }
    }

    RNN_Genome* response_genome = NULL;

    if (local_index >= 0) {
        examm_mutex.lock();
        Island* island = p2p_strategy->get_island_by_index(local_index);
        if (island != NULL && island->size() > 0) {
            response_genome = island->get_best_genome()->copy();
        }
        examm_mutex.unlock();
    }

    if (response_genome == NULL) {
        // Send a minimal genome if we don't have one — should not happen in normal operation
        Log::warning("P2P peer %d: no genome available for island %d, requested by peer %d\n",
                     rank, requested_island, source);
        // Create a minimal placeholder — the requester will handle NULL gracefully
        // We still need to send something to avoid deadlock
        examm_mutex.lock();
        RNN_Genome* seed = p2p_strategy->get_seed_genome();
        if (seed != NULL) {
            response_genome = seed->copy();
        }
        examm_mutex.unlock();
    }

    if (response_genome != NULL) {
        p2p_send_genome(source, response_genome, CROSSOVER_RESPONSE_LENGTH_TAG, CROSSOVER_RESPONSE_TAG);
        delete response_genome;
    }
}

void P2PExamm::replicate_genome(RNN_Genome* genome, int32_t global_island_id) {
    vector<int32_t> successors = ring->get_k_successors(rank, REPLICATION_FACTOR);

    for (int32_t successor : successors) {
        if (successor == rank) continue;

        Log::debug("P2P peer %d: replicating genome to peer %d for island %d\n",
                   rank, successor, global_island_id);

        // Send island ID with its own tag
        int32_t island_msg[1];
        island_msg[0] = global_island_id;
        MPI_Send(island_msg, 1, MPI_INT, successor, REPLICATION_ISLAND_ID_TAG, MPI_COMM_WORLD);

        // Send the genome with length/data tags
        p2p_send_genome(successor, genome, REPLICATION_LENGTH_TAG, REPLICATION_TAG);
    }
}

void P2PExamm::handle_replication(int32_t source) {
    MPI_Status status;

    // First receive the island ID (sent with REPLICATION_ISLAND_ID_TAG)
    int32_t island_msg[1];
    MPI_Recv(island_msg, 1, MPI_INT, source, REPLICATION_ISLAND_ID_TAG, MPI_COMM_WORLD, &status);
    int32_t global_island_id = island_msg[0];

    // Then receive the genome
    RNN_Genome* genome = p2p_receive_genome(source, REPLICATION_LENGTH_TAG, REPLICATION_TAG);

    Log::debug("P2P peer %d: received replica for island %d from peer %d\n",
               rank, global_island_id, source);

    replica_store->store_replica(global_island_id, genome);
    delete genome;
}

void P2PExamm::check_termination() {
    // Estimate global genome count from local count and number of active peers.
    // Each peer evaluates at roughly the same rate, so we can approximate:
    int32_t num_active = ring->get_num_active_peers();
    int32_t estimated_global = local_evaluated_count * num_active;

    Log::debug("P2P peer %d: termination check — local: %d, estimated global: %d, max: %d\n",
               rank, local_evaluated_count, estimated_global, max_genomes);

    // Each peer is responsible for max_genomes / num_peers genomes
    int32_t local_max = (max_genomes > 0) ? (max_genomes / num_active + 1) : 0;

    if (local_max > 0 && local_evaluated_count >= local_max) {
        Log::info("P2P peer %d: local genome count %d >= local_max %d, terminating\n",
                  rank, local_evaluated_count, local_max);
        terminated = true;

        // Notify all other active peers to terminate
        vector<int32_t> active_peers = ring->get_all_active_peers();
        for (int32_t peer : active_peers) {
            if (peer != rank) {
                p2p_send_termination_confirm(peer);
            }
        }
    }
}

void P2PExamm::sync_best_genome() {
    // Log the local best fitness — in a true P2P system, best genome sharing
    // happens opportunistically via replication rather than synchronized allreduce
    examm_mutex.lock();
    double local_best = local_examm->get_best_fitness();
    examm_mutex.unlock();

    Log::info("P2P peer %d: local best fitness: %f\n", rank, local_best);
}

void P2PExamm::send_heartbeats() {
    int32_t successor = ring->get_successor(rank);
    int32_t predecessor = ring->get_predecessor(rank);

    if (successor != rank) {
        p2p_send_heartbeat(successor, rank);
    }
    if (predecessor != rank && predecessor != successor) {
        p2p_send_heartbeat(predecessor, rank);
    }
}

void P2PExamm::handle_heartbeat(int32_t source) {
    int32_t sender = p2p_receive_heartbeat(source);
    last_heartbeat[sender] = std::chrono::steady_clock::now();
    Log::debug("P2P peer %d: received heartbeat from peer %d\n", rank, sender);
}

void P2PExamm::check_peer_liveness() {
    auto now = std::chrono::steady_clock::now();
    int32_t timeout_seconds = HEARTBEAT_INTERVAL_S * HEARTBEAT_TIMEOUT_MULTIPLIER;

    for (auto& pair : last_heartbeat) {
        int32_t peer_id = pair.first;
        if (!ring->is_active(peer_id)) continue;

        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - pair.second).count();
        if (elapsed > timeout_seconds) {
            Log::warning("P2P peer %d: peer %d has not sent heartbeat for %ld seconds, declaring dead\n",
                         rank, peer_id, (long)elapsed);
            handle_peer_failure(peer_id);
        }
    }
}

void P2PExamm::handle_peer_failure(int32_t failed_peer) {
    Log::info("P2P peer %d: handling failure of peer %d\n", rank, failed_peer);

    // Get the islands that the failed peer owned before removing it
    vector<int32_t> failed_islands = ring->get_owned_islands(failed_peer);

    // Remove the peer from the ring (redistributes islands)
    ring->remove_peer(failed_peer);

    // Check which of the failed peer's islands are now assigned to us
    vector<int32_t> my_new_islands;
    for (int32_t island_id : failed_islands) {
        if (ring->get_owner(island_id) == rank) {
            my_new_islands.push_back(island_id);
        }
    }

    if (!my_new_islands.empty()) {
        Log::info("P2P peer %d: taking over %d islands from failed peer %d\n",
                  rank, (int32_t)my_new_islands.size(), failed_peer);

        // Recover island populations from replica store
        for (int32_t island_id : my_new_islands) {
            vector<RNN_Genome*> replicas = replica_store->get_replicas(island_id);
            Log::info("P2P peer %d: recovering island %d with %d replicas\n",
                      rank, island_id, (int32_t)replicas.size());
            // Note: full island recovery would require adding islands to the local strategy
            // at runtime. For now, log the recovery — full dynamic island addition is
            // a future enhancement.
        }
    }

    // Remove from heartbeat tracking
    last_heartbeat.erase(failed_peer);
}
