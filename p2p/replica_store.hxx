#ifndef P2P_REPLICA_STORE_HXX
#define P2P_REPLICA_STORE_HXX

#include <cstdint>
#include <map>
#include <vector>

#include "rnn/rnn_genome.hxx"

using std::map;
using std::vector;

/**
 * Stores replicated genomes from neighboring peers for fault tolerance.
 * When a peer fails, its successor can recover the failed peer's island
 * populations from the replicas stored here.
 */
class ReplicaStore {
   private:
    map<int32_t, vector<RNN_Genome*>> replicas;  // global_island_id -> genomes
    int32_t max_per_island;

   public:
    ReplicaStore(int32_t _max_per_island);
    ~ReplicaStore();

    /**
     * Stores a copy of the genome as a replica for the given island.
     * If the island already has max_per_island replicas, the worst one is evicted.
     */
    void store_replica(int32_t global_island_id, RNN_Genome* genome);

    /**
     * Returns all stored replicas for a given island.
     * These are pointers to internally owned genomes — do not delete them.
     */
    vector<RNN_Genome*> get_replicas(int32_t global_island_id) const;

    /**
     * Removes and frees all replicas for the given island.
     */
    void clear_island(int32_t global_island_id);

    /**
     * Returns the number of replicas stored for a given island.
     */
    int32_t get_replica_count(int32_t global_island_id) const;
};

#endif
