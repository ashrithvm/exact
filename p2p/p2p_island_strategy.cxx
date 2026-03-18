#include "p2p_island_strategy.hxx"

#include "common/log.hxx"

P2PIslandSpeciationStrategy::P2PIslandSpeciationStrategy(
    int32_t _my_rank, PeerRing* _ring, vector<int32_t> _global_island_ids,
    int32_t _number_of_local_islands, int32_t _max_island_size, double _mutation_rate,
    double _intra_island_crossover_rate, double _inter_island_crossover_rate, RNN_Genome* _seed_genome,
    string _island_ranking_method, string _repopulation_method, int32_t _extinction_event_generation_number,
    int32_t _num_mutations, int32_t _islands_to_exterminate, int32_t _max_genomes, bool _repeat_extinction,
    bool _start_filled, bool _transfer_learning, string _transfer_learning_version, int32_t _seed_stirs,
    bool _tl_epigenetic_weights, std::vector<std::string> _possible_node_types, int32_t _is_harada_selection,
    double _harada_selection_ratio, int32_t _is_sweet
)
    : IslandSpeciationStrategy(
          _number_of_local_islands, _max_island_size, _mutation_rate,
          _intra_island_crossover_rate, _inter_island_crossover_rate, _seed_genome,
          _island_ranking_method, _repopulation_method, _extinction_event_generation_number,
          _num_mutations, _islands_to_exterminate, _max_genomes, _repeat_extinction,
          _start_filled, _transfer_learning, _transfer_learning_version, _seed_stirs,
          _tl_epigenetic_weights, _possible_node_types, _is_harada_selection,
          _harada_selection_ratio, _is_sweet
      ),
      my_rank(_my_rank),
      ring(_ring),
      global_island_ids(_global_island_ids) {}

void P2PIslandSpeciationStrategy::set_remote_genome_requester(function<RNN_Genome*(int32_t)> requester) {
    remote_genome_requester = requester;
}

RNN_Genome* P2PIslandSpeciationStrategy::generate_for_filled_island(
    uniform_real_distribution<double>& rng_0_1, minstd_rand0& generator,
    function<void(int32_t, RNN_Genome*)>& mutate, function<RNN_Genome*(RNN_Genome*, RNN_Genome*)>& crossover
) {
    int32_t gen_island = get_generation_island();
    Island* island = get_island_by_index(gen_island);
    RNN_Genome* genome = NULL;
    double r = rng_0_1(generator);

    double mut_rate = get_mutation_rate();
    double intra_rate = get_intra_island_crossover_rate();
    int32_t sweet = get_is_sweet();
    int32_t harada = get_is_harada_selection();
    double harada_ratio = get_harada_selection_ratio();
    int32_t n_mutations = get_num_mutations();

    // ========================================================================
    // 1. MUTATION: INSIDE CURRENT ISLAND ONLY
    // ========================================================================
    if (!island->is_full() || r < mut_rate) {
        Log::debug("P2P: performing mutation on local island %d (global %d)\n",
                   gen_island, global_island_ids[gen_island]);

        if (harada) {
            island->copy_random_genome_harada_selection(rng_0_1, generator, &genome, harada_ratio);
        } else if (sweet && island->evaluating_genomes.size() > 0) {
            island->copy_random_SWEET_genome(rng_0_1, generator, &genome);
        } else {
            island->copy_random_genome(rng_0_1, generator, &genome);
        }
        mutate(n_mutations, genome);

    // ========================================================================
    // 2. INTRA-ISLAND CROSSOVER: BOTH PARENTS INSIDE CURRENT ISLAND
    // ========================================================================
    } else if (r < intra_rate || (number_filled_islands() <= 1 && ring->get_num_active_peers() <= 1)) {
        Log::debug("P2P: performing intra-island crossover on local island %d (global %d)\n",
                   gen_island, global_island_ids[gen_island]);
        RNN_Genome* parent1 = NULL;
        RNN_Genome* parent2 = NULL;

        if (sweet && island->evaluating_genomes.size() > 0 && harada) {
            island->copy_two_SWEET_Harada_genomes(rng_0_1, generator, &parent1, &parent2, harada_ratio);
        } else if (sweet && island->evaluating_genomes.size() > 0) {
            island->copy_two_SWEET_genomes(rng_0_1, generator, &parent1, &parent2);
        } else if (harada) {
            island->copy_two_harada_genomes(rng_0_1, generator, &parent1, &parent2, harada_ratio);
        } else {
            island->copy_two_random_genomes(rng_0_1, generator, &parent1, &parent2);
        }

        genome = crossover(parent1, parent2);

        if (harada) {
            double inherited_freq = (parent1->search_frequency + parent2->search_frequency) / 2.0;
            if (genome != NULL) genome->search_frequency = inherited_freq;
        }

        delete parent1;
        delete parent2;

    // ========================================================================
    // 3. INTER-ISLAND CROSSOVER: CROSSING BORDERS (LOCAL OR REMOTE)
    // ========================================================================
    } else {
        Log::debug("P2P: performing inter-island crossover from local island %d (global %d)\n",
                   gen_island, global_island_ids[gen_island]);

        // Get parent1 from the current local island
        RNN_Genome* parent1 = NULL;
        if (sweet && island->evaluating_genomes.size() > 0 && harada) {
            island->copy_random_SWEET_Harada_genome(rng_0_1, generator, &parent1, harada_ratio);
        } else if (sweet && island->evaluating_genomes.size() > 0) {
            island->copy_random_SWEET_genome(rng_0_1, generator, &parent1);
        } else if (harada) {
            island->copy_random_genome_harada_selection(rng_0_1, generator, &parent1, harada_ratio);
        } else {
            island->copy_random_genome(rng_0_1, generator, &parent1);
        }

        // Get parent2 from another island — try local first, then remote
        RNN_Genome* parent2 = NULL;

        if (number_filled_islands() > 1) {
            // We have other full local islands — use one
            int32_t other_local = get_other_full_island(rng_0_1, generator, gen_island);
            Island* other_island = get_island_by_index(other_local);

            if (sweet && other_island->evaluating_genomes.size() > 0 && harada) {
                other_island->copy_random_SWEET_Harada_genome(rng_0_1, generator, &parent2, harada_ratio);
            } else if (sweet && other_island->evaluating_genomes.size() > 0) {
                other_island->copy_random_SWEET_genome(rng_0_1, generator, &parent2);
            } else if (harada) {
                other_island->copy_random_genome_harada_selection(rng_0_1, generator, &parent2, harada_ratio);
            } else {
                parent2 = other_island->get_best_genome()->copy();
            }

            Log::debug("P2P: inter-island crossover with local island %d (global %d)\n",
                       other_local, global_island_ids[other_local]);
        } else if (remote_genome_requester) {
            // No other full local islands — request from a remote peer
            int32_t total_islands = ring->get_num_islands();

            // Collect all remote islands
            vector<int32_t> remote_islands;
            for (int32_t i = 0; i < total_islands; i++) {
                if (ring->get_owner(i) != my_rank) {
                    remote_islands.push_back(i);
                }
            }

            if (!remote_islands.empty()) {
                int32_t remote_island_id = remote_islands[(int32_t)(rng_0_1(generator) * remote_islands.size())];
                Log::info("P2P: requesting genome from remote island %d (owned by peer %d)\n",
                          remote_island_id, ring->get_owner(remote_island_id));
                parent2 = remote_genome_requester(remote_island_id);
            }
        }

        if (parent2 == NULL) {
            // Fallback: mutation if we couldn't get a second parent
            Log::debug("P2P: falling back to mutation (no second parent available)\n");
            mutate(n_mutations, parent1);
            genome = parent1;
            if (genome != NULL && genome->outputs_unreachable()) {
                delete genome;
                genome = NULL;
            }
            return genome;
        }

        // Swap so the first parent is the more fit parent
        if (parent1->get_fitness() > parent2->get_fitness()) {
            RNN_Genome* tmp = parent1;
            parent1 = parent2;
            parent2 = tmp;
        }

        genome = crossover(parent1, parent2);

        if (harada && genome != NULL) {
            double inherited_freq = (parent1->search_frequency + parent2->search_frequency) / 2.0;
            genome->search_frequency = inherited_freq;
        }

        delete parent1;
        delete parent2;
    }

    if (genome != NULL && genome->outputs_unreachable()) {
        delete genome;
        genome = NULL;
    }
    return genome;
}

RNN_Genome* P2PIslandSpeciationStrategy::generate_genome(
    uniform_real_distribution<double>& rng_0_1, minstd_rand0& generator,
    function<void(int32_t, RNN_Genome*)>& mutate, function<RNN_Genome*(RNN_Genome*, RNN_Genome*)>& crossover
) {
    // Call parent's generate_genome which handles island round-robin,
    // initialization, repopulation, and calls our overridden generate_for_filled_island
    RNN_Genome* genome = IslandSpeciationStrategy::generate_genome(rng_0_1, generator, mutate, crossover);

    if (genome != NULL) {
        // Remap local group_id to global island ID
        int32_t local_id = genome->get_group_id();
        if (local_id >= 0 && local_id < (int32_t)global_island_ids.size()) {
            genome->set_group_id(global_island_ids[local_id]);
        }
    }

    return genome;
}

int32_t P2PIslandSpeciationStrategy::local_to_global_island(int32_t local_index) const {
    if (local_index < 0 || local_index >= (int32_t)global_island_ids.size()) {
        return -1;
    }
    return global_island_ids[local_index];
}

vector<int32_t> P2PIslandSpeciationStrategy::get_global_island_ids() const {
    return global_island_ids;
}
