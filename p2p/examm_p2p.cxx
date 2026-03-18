#include <chrono>
#include <iomanip>
#include <iostream>
using std::fixed;
using std::setprecision;
using std::setw;

#include <string>
using std::string;

#include <vector>
using std::vector;

#include "common/log.hxx"
#include "common/process_arguments.hxx"
#include "examm/examm.hxx"
#include "rnn/genome_property.hxx"
#include "mpi.h"
#include "p2p_examm.hxx"
#include "p2p_island_strategy.hxx"
#include "p2p_tags.hxx"
#include "peer_ring.hxx"
#include "rnn/generate_nn.hxx"
#include "time_series/time_series.hxx"
#include "weights/weight_rules.hxx"
#include "weights/weight_update.hxx"

vector<string> arguments;

vector<vector<vector<double>>> training_inputs;
vector<vector<vector<double>>> training_outputs;
vector<vector<vector<double>>> validation_inputs;
vector<vector<vector<double>>> validation_outputs;

int main(int argc, char** argv) {
    std::cout << "P2P EXAMM starting up!" << std::endl;

    // P2P requires MPI_THREAD_MULTIPLE because we use MPI from two threads:
    // Thread 1 (evolution): calls request_remote_genome which does MPI send/recv
    // Thread 2 (handler): probes and receives incoming MPI messages
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided < MPI_THREAD_MULTIPLE) {
        std::cerr << "ERROR: MPI implementation does not support MPI_THREAD_MULTIPLE "
                  << "(provided level: " << provided << "). "
                  << "P2P EXAMM requires multi-threaded MPI support." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int32_t rank, num_peers;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_peers);
    std::cout << "P2P peer " << rank << " of " << num_peers << std::endl;

    arguments = vector<string>(argv, argv + argc);

    Log::initialize(arguments);
    Log::set_rank(rank);
    Log::set_id("p2p_main_" + std::to_string(rank));

    // All peers load the same training/validation data
    TimeSeriesSets* time_series_sets = TimeSeriesSets::generate_from_arguments(arguments);
    get_train_validation_data(
        arguments, time_series_sets, training_inputs, training_outputs, validation_inputs, validation_outputs
    );

    WeightUpdate* weight_update_method = new WeightUpdate();
    weight_update_method->generate_from_arguments(arguments);

    WeightRules* weight_rules = new WeightRules();
    weight_rules->initialize_from_args(arguments);

    RNN_Genome* seed_genome = get_seed_genome(arguments, time_series_sets, weight_rules);

    // Parse island configuration
    int32_t island_size;
    get_argument(arguments, "--island_size", true, island_size);
    int32_t number_islands;
    get_argument(arguments, "--number_islands", true, number_islands);
    int32_t max_genomes = 0;
    get_argument(arguments, "--max_genomes", false, max_genomes);
    int32_t is_sweet = 0;
    get_argument(arguments, "--is_sweet", false, is_sweet);

    // Create the peer ring for island assignment
    PeerRing* ring = new PeerRing(num_peers, number_islands);

    // Determine which islands this peer owns
    vector<int32_t> my_islands = ring->get_owned_islands(rank);
    int32_t num_local_islands = (int32_t)my_islands.size();

    Log::info("P2P peer %d: owns %d islands:", rank, num_local_islands);
    for (int32_t id : my_islands) {
        Log::info_no_header(" %d", id);
    }
    Log::info_no_header("\n");

    if (num_local_islands == 0) {
        Log::fatal("P2P peer %d: no islands assigned! Increase --number_islands or decrease number of peers.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Parse speciation strategy parameters (same as generate_island_speciation_strategy_from_arguments)
    int32_t extinction_event_generation_number = 0;
    get_argument(arguments, "--extinction_event_generation_number", false, extinction_event_generation_number);
    int32_t islands_to_exterminate = 0;
    get_argument(arguments, "--islands_to_exterminate", false, islands_to_exterminate);
    string island_ranking_method = "";
    get_argument(arguments, "--island_ranking_method", false, island_ranking_method);
    string repopulation_method = "";
    get_argument(arguments, "--repopulation_method", false, repopulation_method);
    int32_t num_mutations = 1;
    get_argument(arguments, "--num_mutations", false, num_mutations);
    int32_t is_harada_selection = 0;
    get_argument(arguments, "--is_harada_selection", false, is_harada_selection);
    double harada_selection_ratio = 0.0;
    get_argument(arguments, "--harada_selection_ratio", false, harada_selection_ratio);

    double mutation_rate = 0.70, intra_island_co_rate = 0.20, inter_island_co_rate = 0.10;
    if (num_local_islands == 1) {
        // If this peer has only one local island, inter-island crossover goes remote
        // Keep the rates as-is; the P2P strategy handles remote crossover
    }

    bool repeat_extinction = argument_exists(arguments, "--repeat_extinction");
    string transfer_learning_version = "";
    bool transfer_learning = get_argument(arguments, "--transfer_learning_version", false, transfer_learning_version);
    int32_t seed_stirs = 0;
    get_argument(arguments, "--seed_stirs", false, seed_stirs);
    bool start_filled = argument_exists(arguments, "--start_filled");
    bool tl_epigenetic_weights = argument_exists(arguments, "--tl_epigenetic_weights");

    vector<string> possible_node_types;
    get_argument_vector(arguments, "--possible_node_types", false, possible_node_types);

    // Create the P2P island speciation strategy with only local islands
    // max_genomes for local strategy: each peer's share (approximate)
    int32_t local_max_genomes = (max_genomes > 0) ? (max_genomes / num_peers + 1) : 0;

    P2PIslandSpeciationStrategy* p2p_strategy = new P2PIslandSpeciationStrategy(
        rank, ring, my_islands,
        num_local_islands, island_size, mutation_rate, intra_island_co_rate, inter_island_co_rate,
        seed_genome, island_ranking_method, repopulation_method, extinction_event_generation_number,
        num_mutations, islands_to_exterminate, local_max_genomes, repeat_extinction,
        start_filled, transfer_learning, transfer_learning_version, seed_stirs,
        tl_epigenetic_weights, possible_node_types, is_harada_selection, harada_selection_ratio, is_sweet
    );

    // Parse remaining EXAMM configuration
    int64_t max_wallclock_seconds = 0;
    get_argument(arguments, "--max_wallclock_seconds", false, max_wallclock_seconds);
    string output_directory = "";
    get_argument(arguments, "--output_directory", false, output_directory);

    // Add rank suffix to output directory so peers don't conflict
    if (!output_directory.empty()) {
        output_directory += "/peer_" + std::to_string(rank);
    }

    string save_genome_option = "all_best_genomes";
    get_argument(arguments, "--save_genome_option", false, save_genome_option);
    int32_t growth_phase_genomes = 0;
    get_argument(arguments, "--growth_phase_genomes", false, growth_phase_genomes);
    int32_t reduction_phase_genomes = 0;
    get_argument(arguments, "--reduction_phase_genomes", false, reduction_phase_genomes);
    bool generate_op_log = false;
    get_argument(arguments, "--generate_op_log", false, generate_op_log);
    bool generate_visualization_json = false;
    get_argument(arguments, "--generate_visualization_json", false, generate_visualization_json);
    int32_t genome_size_log = 0;
    get_argument(arguments, "--genome_size_log", false, genome_size_log);

    // Create and configure GenomeProperty from arguments
    GenomeProperty* genome_property = new GenomeProperty();
    genome_property->generate_genome_property_from_arguments(arguments);
    genome_property->get_time_series_parameters(time_series_sets);

    // Create the local EXAMM instance with only local islands
    EXAMM* local_examm = new EXAMM(
        island_size, num_local_islands, local_max_genomes, max_wallclock_seconds,
        p2p_strategy, weight_rules, genome_property,
        output_directory, save_genome_option, generate_op_log, generate_visualization_json,
        growth_phase_genomes, reduction_phase_genomes, genome_size_log,
        is_harada_selection, harada_selection_ratio, is_sweet
    );

    if (possible_node_types.size() > 0) {
        local_examm->set_possible_node_types(possible_node_types);
    }

    // Set innovation number offsets for global uniqueness
    local_examm->set_innovation_offsets(rank * INNOVATION_RANGE_SIZE, rank * INNOVATION_RANGE_SIZE);

    if (rank == 0) {
        write_time_series_to_file(arguments, time_series_sets);
    }

    // Create and run the P2P coordinator
    P2PExamm p2p_examm(
        rank, num_peers, local_examm, p2p_strategy, ring,
        max_genomes, is_sweet,
        training_inputs, training_outputs, validation_inputs, validation_outputs,
        weight_update_method
    );

    // Barrier to ensure all peers are ready before starting
    MPI_Barrier(MPI_COMM_WORLD);

    p2p_examm.run();

    Log::set_id("p2p_main_" + std::to_string(rank));
    Log::info("P2P peer %d completed!\n", rank);
    Log::release_id("p2p_main_" + std::to_string(rank));

    MPI_Finalize();

    delete time_series_sets;
    // local_examm owns the strategy, so don't double-delete
    delete local_examm;
    delete ring;

    return 0;
}
