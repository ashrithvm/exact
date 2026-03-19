#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "weights/weight_rules.hxx"
#include "rnn/generate_nn.hxx"
#include "rnn/rnn_genome.hxx"
#include "examm/island_speciation_strategy.hxx"
#include "examm/examm.hxx"
#include "rnn/genome_property.hxx"

#include "mpi.h"

// Pull in the P2P helper implementations from the production MPI file.
// This keeps the tests coupled to the exact code path without needing to
// refactor the helper logic into a separate library.
#define EXAMM_MPI_P2P_UNIT_TEST
#include "mpi/examm_mpi.cxx"

static void require(bool cond, const char* msg) {
    if (!cond) {
        std::cerr << "TEST FAILED: " << msg << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}

static RNN_Genome* make_ff_genome(
    WeightRules* weight_rules,
    const std::vector<std::string>& inputs,
    int32_t hidden_layers,
    int32_t hidden_nodes,
    const std::vector<std::string>& outputs,
    int32_t max_recurrent_depth
) {
    // `create_ff` is a macro defined in rnn/generate_nn.hxx.
    RNN_Genome* genome = create_ff(inputs, hidden_layers, hidden_nodes, outputs, max_recurrent_depth, weight_rules);
    genome->set_group_id(0);
    genome->set_generation_id(1);
    return genome;
}

static EXAMM* make_minimal_examm_for_injection_test(
    WeightRules* weight_rules,
    RNN_Genome* seed_genome
) {
    GenomeProperty* genome_property = new GenomeProperty();

    // IslandSpeciationStrategy constructor expects many args; for injection
    // we keep extinction/repopulation disabled and start_filled=false.
    IslandSpeciationStrategy* speciation = new IslandSpeciationStrategy(
        /* number_of_islands */ 1,
        /* max_island_size */ 8,
        /* mutation_rate */ 0.7,
        /* intra_island_crossover_rate */ 0.2,
        /* inter_island_crossover_rate */ 0.1,
        /* seed_genome */ seed_genome,
        /* island_ranking_method */ "",
        /* repopulation_method */ "",
        /* extinction_event_generation_number */ 0,
        /* num_mutations */ 1,
        /* islands_to_exterminate */ 0,
        /* max_genomes */ 0,
        /* repeat_extinction */ false,
        /* start_filled */ false,
        /* transfer_learning */ false,
        /* transfer_learning_version */ "",
        /* seed_stirs */ 0,
        /* tl_epigenetic_weights */ false,
        /* possible_node_types */ {},
        /* is_harada_selection */ 0,
        /* harada_selection_ratio */ 0.0,
        /* is_sweet */ 0
    );

    EXAMM* ex = new EXAMM(
        /* island_size */ 8,
        /* number_islands */ 1,
        /* max_genomes */ 0,
        /* max_wallclock_seconds */ 0,
        /* speciation_strategy */ speciation,
        /* weight_rules */ weight_rules,
        /* genome_property */ genome_property,
        /* output_directory */ "",
        /* save_genome_option */ "",
        /* generate_op_log */ false,
        /* generate_visualization_json */ false,
        /* growth_phase_genomes */ 0,
        /* reduction_phase_genomes */ 0,
        /* genome_size_log */ 0,
        /* is_harada_selection */ 0,
        /* harada_selection_ratio */ 0.0,
        /* is_sweet */ 0
    );
    return ex;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int32_t rank = 0, max_rank = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &max_rank);

    // This test assumes exactly 2 ranks for the P2P send/recv chunk test.
    require(max_rank == 2, "run with: mpirun -np 2 mpi/test_examm_mpi_p2p_helpers");

    // ---------------------------------------------------------------------
    // 1) stable_hash_fnv1a_64
    // ---------------------------------------------------------------------
    if (rank == 0) {
        const uint64_t abc_hash_expected = 0xe71fa2190541574bULL; // FNV-1a 64-bit of "abc"
        uint64_t abc_hash = stable_hash_fnv1a_64("abc");
        require(abc_hash == abc_hash_expected, "stable_hash_fnv1a_64('abc') matches expected constant");
        uint64_t empty_hash = stable_hash_fnv1a_64("");
        require(empty_hash == 0xcbf29ce484222325ULL, "stable_hash_fnv1a_64('') matches expected constant");
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // ---------------------------------------------------------------------
    // 2) with_peer_output_directory
    // ---------------------------------------------------------------------
    {
        std::vector<std::string> args = {"prog", "--output_directory", "./base"};
        std::vector<std::string> out = with_peer_output_directory(args, "./base/p2p_rank_1");
        require(out.size() >= 3, "with_peer_output_directory size");
        require(out[2] == "./base/p2p_rank_1", "with_peer_output_directory replaces output_directory value");
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // ---------------------------------------------------------------------
    // 3) genome_owner_rank deterministic ownership
    // ---------------------------------------------------------------------
    if (rank == 0) {
        WeightRules* wr = new WeightRules();
        std::vector<std::string> inputs = {"in1", "in2"};
        std::vector<std::string> outputs = {"out1", "out2"};
        RNN_Genome* g = make_ff_genome(wr, inputs, 1, 2, outputs, 0);

        int32_t owner = genome_owner_rank(g, /*max_rank*/ 2);
        require(owner >= 0 && owner < 2, "genome_owner_rank returns rank in range");

        std::string structural_hash = g->get_structural_hash();
        uint64_t expected_h = stable_hash_fnv1a_64(structural_hash);
        int32_t expected_owner = static_cast<int32_t>(expected_h % static_cast<uint64_t>(2));
        require(owner == expected_owner, "genome_owner_rank equals hash(structural_hash) % max_rank");

        delete g;
        delete wr;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // ---------------------------------------------------------------------
    // 4) Chunked P2P send/receive + progress helpers
    // ---------------------------------------------------------------------
    // We use BACKUP kind to ensure progress_incoming always inserts (it
    // bypasses ownership filtering in progress_incoming()).
    WeightRules* wr = new WeightRules();
    std::vector<std::string> inputs = {"in1", "in2"};
    std::vector<std::string> outputs = {"out1", "out2"};

    // Build a genome large enough to exceed the 32KB chunk size.
    // (If this ends up being too small for your build, increase hidden_nodes.)
    int32_t hidden_layers = 4;
    int32_t hidden_nodes = 16;
    int32_t max_recurrent_depth = 2;
    RNN_Genome* big = make_ff_genome(wr, inputs, hidden_layers, hidden_nodes, outputs, max_recurrent_depth);
    big->set_generation_id(123);
    big->set_group_id(0);

    const int32_t chunk_size = 32768;
    int32_t big_length = 0;
    if (rank == 0) {
        char* arr = nullptr;
        big->write_to_array(&arr, big_length);
        require(big_length > chunk_size, "serialized genome length exceeds 32KB chunk size");
        free(arr);
    }
    MPI_Bcast(&big_length, 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<OutgoingGenomeTransfer> pending_outgoing;
    std::vector<IncomingGenomeTransfer> pending_incoming;

    if (rank == 0) {
        queue_genome_send(GenomeTransferKind::BACKUP, /*dest*/ 1, big, pending_outgoing);
    } else {
        // Receive the header length first.
        MPI_Status st;
        int32_t length = 0;
        MPI_Recv(&length, 1, MPI_INT, /*source*/ 0, BACKUP_GENOME_TAG, MPI_COMM_WORLD, &st);
        require(length == big_length, "receiver length header matches expected length");
        post_genome_receive(GenomeTransferKind::BACKUP, /*source*/ 0, length, pending_incoming);
    }

    // Create minimal EXAMM on the receiver so progress_incoming() can inject.
    EXAMM* ex_for_injection = nullptr;
    IslandSpeciationStrategy* spec_strat = nullptr;
    if (rank == 1) {
        // Seed genome can be small; it just needs valid structure for EXAMM initialization.
        RNN_Genome* seed = make_ff_genome(wr, inputs, 0, 0, outputs, 0);
        seed->set_generation_id(0);
        seed->set_group_id(0);

        // Create the examm with a speciation strategy we can inspect.
        GenomeProperty* gp = new GenomeProperty();
        spec_strat = new IslandSpeciationStrategy(
            /* number_of_islands */ 1,
            /* max_island_size */ 8,
            /* mutation_rate */ 0.7,
            /* intra_island_crossover_rate */ 0.2,
            /* inter_island_crossover_rate */ 0.1,
            /* seed_genome */ seed,
            /* island_ranking_method */ "",
            /* repopulation_method */ "",
            /* extinction_event_generation_number */ 0,
            /* num_mutations */ 1,
            /* islands_to_exterminate */ 0,
            /* max_genomes */ 0,
            /* repeat_extinction */ false,
            /* start_filled */ false,
            /* transfer_learning */ false,
            /* transfer_learning_version */ "",
            /* seed_stirs */ 0,
            /* tl_epigenetic_weights */ false,
            /* possible_node_types */ {},
            /* is_harada_selection */ 0,
            /* harada_selection_ratio */ 0.0,
            /* is_sweet */ 0
        );

        ex_for_injection = new EXAMM(
            /* island_size */ 8,
            /* number_islands */ 1,
            /* max_genomes */ 0,
            /* max_wallclock_seconds */ 0,
            /* speciation_strategy */ spec_strat,
            /* weight_rules */ wr,
            /* genome_property */ gp,
            /* output_directory */ "",
            /* save_genome_option */ "",
            /* generate_op_log */ false,
            /* generate_visualization_json */ false,
            /* growth_phase_genomes */ 0,
            /* reduction_phase_genomes */ 0,
            /* genome_size_log */ 0,
            /* is_harada_selection */ 0,
            /* harada_selection_ratio */ 0.0,
            /* is_sweet */ 0
        );
    }

    int32_t evaluated_before = 0;
    if (rank == 1) {
        evaluated_before = spec_strat->get_evaluated_genomes();
    }

    // Progress outgoing (rank 0) and incoming (rank 1) until completion.
    while (true) {
        if (rank == 0) {
            progress_outgoing(pending_outgoing);
        } else {
            progress_incoming(pending_incoming, ex_for_injection, rank, /*max_rank*/ 2);
        }

        bool outgoing_done = (rank == 0) ? pending_outgoing.empty() : true;
        bool incoming_done = (rank == 1) ? pending_incoming.empty() : true;
        int local_done = outgoing_done && incoming_done;
        int all_done = 0;
        MPI_Allreduce(&local_done, &all_done, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
        if (all_done) break;

        // Avoid tight spin.
        if (rank == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    if (rank == 1) {
        int32_t evaluated_after = spec_strat->get_evaluated_genomes();
        require(evaluated_after == evaluated_before + 1, "progress_incoming injected exactly one migrated/backup genome");
    }

    // ---------------------------------------------------------------------
    // 5) broadcast_genome_seed
    // ---------------------------------------------------------------------
    // Broadcast a small genome from rank 0 to rank 1 and verify structural hash matches.
    RNN_Genome* seed = nullptr;
    if (rank == 0) {
        seed = make_ff_genome(wr, inputs, 0, 0, outputs, 0);
        seed->set_generation_id(0);
        seed->set_group_id(0);
    }

    // NOTE: broadcast_genome_seed will return `seed` on rank 0 and a newly allocated genome on other ranks.
    RNN_Genome* broadcasted = broadcast_genome_seed(seed, rank, /*max_rank*/ 2);

    uint64_t seed_struct_hash_fnv = 0;
    if (broadcasted != nullptr) {
        seed_struct_hash_fnv = stable_hash_fnv1a_64(broadcasted->get_structural_hash());
    }
    MPI_Bcast(&seed_struct_hash_fnv, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

    uint64_t broadcasted_struct_hash_fnv = 0;
    if (broadcasted != nullptr) {
        broadcasted_struct_hash_fnv = stable_hash_fnv1a_64(broadcasted->get_structural_hash());
    }
    require(broadcasted_struct_hash_fnv == seed_struct_hash_fnv, "broadcast_genome_seed preserves genome structural hash");

    if (rank == 0) {
        delete broadcasted;  // `seed` is owned by us here.
    } else {
        delete broadcasted;
    }

    delete big;
    delete wr;

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 1 && ex_for_injection != nullptr) {
        delete ex_for_injection;
    }

    MPI_Finalize();
    return 0;
}

