#include <cassert>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "common/log.hxx"
#include "examm/examm.hxx"
#include "examm/island_speciation_strategy.hxx"
#include "rnn/generate_nn.hxx"
#include "rnn/genome_property.hxx"
#include "rnn/rnn_genome.hxx"
#include "weights/weight_rules.hxx"

static void require(bool cond, const char* msg) {
    if (!cond) {
        std::cerr << "TEST FAILED: " << msg << std::endl;
        std::exit(1);
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
    RNN_Genome* genome = create_ff(inputs, hidden_layers, hidden_nodes, outputs, max_recurrent_depth, weight_rules);
    genome->set_group_id(0);
    genome->set_generation_id(1);
    return genome;
}

int main() {
    // Log requires these arguments. Keep output local to the repo.
    std::vector<std::string> arguments = {
        "test_examm_inject_migrated_genome",
        "--std_message_level",
        "0",
        "--file_message_level",
        "0",
        "--output_directory",
        "./test_output/examm_inject_migrated_genome_logs"
    };
    Log::initialize(arguments);
    Log::set_id("main");

    WeightRules* wr = new WeightRules();
    GenomeProperty* gp = new GenomeProperty();

    std::vector<std::string> inputs = {"in1", "in2"};
    std::vector<std::string> outputs = {"out1", "out2"};

    // Seed genome: minimal valid structure for EXAMM initialization.
    RNN_Genome* seed = make_ff_genome(wr, inputs, 0, 0, outputs, 0);
    seed->set_generation_id(0);
    seed->set_group_id(0);

    IslandSpeciationStrategy* strat = new IslandSpeciationStrategy(
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

    EXAMM* ex = new EXAMM(
        /* island_size */ 8,
        /* number_islands */ 1,
        /* max_genomes */ 0,
        /* max_wallclock_seconds */ 0,
        /* speciation_strategy */ strat,
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

    int32_t evaluated_before = strat->get_evaluated_genomes();

    RNN_Genome* migrated = make_ff_genome(wr, inputs, 1, 2, outputs, 0);
    migrated->set_generation_id(10);
    migrated->set_group_id(0);
    migrated->set_bp_iterations(1);

    bool inserted = ex->inject_migrated_genome(migrated);
    delete migrated;

    int32_t evaluated_after = strat->get_evaluated_genomes();

    require(inserted, "inject_migrated_genome returned true for a genome inserted into empty island");
    require(evaluated_after == evaluated_before + 1, "inject_migrated_genome increments evaluated_genomes by 1");
    require(strat->get_best_genome() != nullptr, "global best genome exists after insertion");

    // EXAMM destructor deletes `wr` (weight_rules) and `gp` (genome_property).
    // Leave the remaining objects to their owning strategies.
    delete ex;

    std::cout << "ALL TESTS PASSED" << std::endl;
    return 0;
}

