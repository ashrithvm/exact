#ifndef P2P_TAGS_HXX
#define P2P_TAGS_HXX

// MPI message tags for P2P communication
// Tags 1-4 are reserved by the original master-worker protocol

#define CROSSOVER_REQUEST_TAG          10
#define CROSSOVER_RESPONSE_LENGTH_TAG  11
#define CROSSOVER_RESPONSE_TAG         12
#define REPLICATION_LENGTH_TAG         13
#define REPLICATION_TAG                14
#define HEARTBEAT_TAG                  15
#define TERMINATION_VOTE_TAG           16
#define TERMINATION_CONFIRM_TAG        17
#define REPLICATION_ISLAND_ID_TAG      18
#define BEST_GENOME_SHARE_LENGTH_TAG   19
#define BEST_GENOME_SHARE_TAG          20

// Each peer gets a non-overlapping range of innovation numbers
// to ensure global uniqueness without communication
#define INNOVATION_RANGE_SIZE  1000000

// Number of successor peers to replicate genomes to
#define REPLICATION_FACTOR  2

// Seconds between heartbeat messages
#define HEARTBEAT_INTERVAL_S  5

// Number of genome evaluations between distributed sync operations
// (termination check and best genome sharing)
#define SYNC_INTERVAL  10

// Multiplier for heartbeat timeout (declare dead after this many missed heartbeats)
#define HEARTBEAT_TIMEOUT_MULTIPLIER  3

#endif
