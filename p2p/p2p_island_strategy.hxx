#ifndef P2P_ISLAND_STRATEGY_HXX
#define P2P_ISLAND_STRATEGY_HXX

#include <cstdint>
#include <functional>
#include <random>
#include <vector>

#include "examm/island_speciation_strategy.hxx"
#include "peer_ring.hxx"
#include "rnn/rnn_genome.hxx"

using std::function;
using std::minstd_rand0;
using std::uniform_real_distribution;
using std::vector;

/**
 * P2P extension of IslandSpeciationStrategy that handles cross-peer
 * inter-island crossover. Each peer only owns a subset of all islands.
 * When inter-island crossover requires a genome from a remote peer's island,
 * this class uses a callback to request it via MPI.
 */
class P2PIslandSpeciationStrategy : public IslandSpeciationStrategy {
   private:
    int32_t my_rank;
    PeerRing* ring;
    vector<int32_t> global_island_ids;  // maps local index -> global island ID

    // Callback set by P2PExamm to request a genome from a remote peer's island.
    // Takes global_island_id, returns a genome copy (caller owns it).
    function<RNN_Genome*(int32_t global_island_id)> remote_genome_requester;

   public:
    P2PIslandSpeciationStrategy(
        int32_t _my_rank, PeerRing* _ring, vector<int32_t> _global_island_ids,
        int32_t _number_of_local_islands, int32_t _max_island_size, double _mutation_rate,
        double _intra_island_crossover_rate, double _inter_island_crossover_rate, RNN_Genome* _seed_genome,
        string _island_ranking_method, string _repopulation_method, int32_t _extinction_event_generation_number,
        int32_t _num_mutations, int32_t _islands_to_exterminate, int32_t _max_genomes, bool _repeat_extinction,
        bool _start_filled, bool _transfer_learning, string _transfer_learning_version, int32_t _seed_stirs,
        bool _tl_epigenetic_weights, std::vector<std::string> _possible_node_types, int32_t _is_harada_selection,
        double _harada_selection_ratio, int32_t _is_sweet
    );

    /**
     * Set the callback for requesting genomes from remote peers.
     * Must be called before any genome generation.
     */
    void set_remote_genome_requester(function<RNN_Genome*(int32_t)> requester);

    /**
     * Override to handle inter-island crossover across peers.
     * If all local islands are not full enough for inter-island crossover,
     * requests a genome from a remote peer's island.
     */
    RNN_Genome* generate_for_filled_island(
        uniform_real_distribution<double>& rng_0_1, minstd_rand0& generator,
        function<void(int32_t, RNN_Genome*)>& mutate, function<RNN_Genome*(RNN_Genome*, RNN_Genome*)>& crossover
    ) override;

    /**
     * Override to remap local group_id to global island ID after generation.
     */
    RNN_Genome* generate_genome(
        uniform_real_distribution<double>& rng_0_1, minstd_rand0& generator,
        function<void(int32_t, RNN_Genome*)>& mutate, function<RNN_Genome*(RNN_Genome*, RNN_Genome*)>& crossover
    ) override;

    /**
     * Map a local island index to the global island ID.
     */
    int32_t local_to_global_island(int32_t local_index) const;

    /**
     * Get all global island IDs owned by this peer.
     */
    vector<int32_t> get_global_island_ids() const;
};

#endif
