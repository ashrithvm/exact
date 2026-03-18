#ifndef P2P_COMM_HXX
#define P2P_COMM_HXX

#include <cstdint>

#include "mpi.h"
#include "rnn/rnn_genome.hxx"

// Send a genome to a target peer using configurable MPI tags.
// Uses 32KB chunked sends to bypass cluster message size limits.
void p2p_send_genome(int32_t target, RNN_Genome* genome, int32_t length_tag, int32_t data_tag);

// Receive a genome from a source peer using configurable MPI tags.
// Handles 32KB chunked receives.
RNN_Genome* p2p_receive_genome(int32_t source, int32_t length_tag, int32_t data_tag);

// Send a crossover request to a remote peer, specifying which global island ID
// we want a genome from.
void p2p_send_crossover_request(int32_t target, int32_t global_island_id);

// Receive a crossover request from a source peer.
// Returns the requested global island ID.
int32_t p2p_receive_crossover_request(int32_t source);

// Send a heartbeat message to a target peer.
void p2p_send_heartbeat(int32_t target, int32_t my_rank);

// Receive a heartbeat message from a source peer.
// Returns the sender's rank.
int32_t p2p_receive_heartbeat(int32_t source);

// Send a termination vote with local evaluated count.
void p2p_send_termination_vote(int32_t target, int32_t local_count);

// Receive a termination vote.
// Returns the sender's local evaluated count.
int32_t p2p_receive_termination_vote(int32_t source);

// Send a termination confirmation to all peers.
void p2p_send_termination_confirm(int32_t target);

// Receive a termination confirmation.
void p2p_receive_termination_confirm(int32_t source);

#endif
