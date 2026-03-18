#include "p2p_comm.hxx"

#include "common/log.hxx"
#include "p2p_tags.hxx"

void p2p_send_genome(int32_t target, RNN_Genome* genome, int32_t length_tag, int32_t data_tag) {
    char* byte_array;
    int32_t length;

    genome->write_to_array(&byte_array, length);

    Log::debug("P2P: sending genome of length: %d to: %d\n", length, target);

    // Send the total length first
    int32_t length_message[1];
    length_message[0] = length;
    MPI_Send(length_message, 1, MPI_INT, target, length_tag, MPI_COMM_WORLD);

    // Send data in 32KB chunks
    int32_t offset = 0;
    int32_t chunk_size = 32768;

    while (offset < length) {
        int32_t send_size = length - offset;
        if (send_size > chunk_size) {
            send_size = chunk_size;
        }

        MPI_Send(byte_array + offset, send_size, MPI_CHAR, target, data_tag, MPI_COMM_WORLD);
        offset += send_size;
    }

    free(byte_array);
}

RNN_Genome* p2p_receive_genome(int32_t source, int32_t length_tag, int32_t data_tag) {
    MPI_Status status;
    int32_t length_message[1];

    // Receive the total length first
    MPI_Recv(length_message, 1, MPI_INT, source, length_tag, MPI_COMM_WORLD, &status);
    int32_t length = length_message[0];

    Log::debug("P2P: receiving genome of length: %d from: %d\n", length, source);

    // Allocate memory for the full message
    char* genome_str = new char[length + 1];

    // Receive data in 32KB chunks
    int32_t offset = 0;
    int32_t chunk_size = 32768;

    while (offset < length) {
        int32_t recv_size = length - offset;
        if (recv_size > chunk_size) {
            recv_size = chunk_size;
        }

        MPI_Recv(genome_str + offset, recv_size, MPI_CHAR, source, data_tag, MPI_COMM_WORLD, &status);
        offset += recv_size;
    }

    genome_str[length] = '\0';

    RNN_Genome* genome = new RNN_Genome(genome_str, length);

    delete[] genome_str;
    return genome;
}

void p2p_send_crossover_request(int32_t target, int32_t global_island_id) {
    int32_t message[1];
    message[0] = global_island_id;
    MPI_Send(message, 1, MPI_INT, target, CROSSOVER_REQUEST_TAG, MPI_COMM_WORLD);
}

int32_t p2p_receive_crossover_request(int32_t source) {
    MPI_Status status;
    int32_t message[1];
    MPI_Recv(message, 1, MPI_INT, source, CROSSOVER_REQUEST_TAG, MPI_COMM_WORLD, &status);
    return message[0];
}

void p2p_send_heartbeat(int32_t target, int32_t my_rank) {
    int32_t message[1];
    message[0] = my_rank;
    MPI_Send(message, 1, MPI_INT, target, HEARTBEAT_TAG, MPI_COMM_WORLD);
}

int32_t p2p_receive_heartbeat(int32_t source) {
    MPI_Status status;
    int32_t message[1];
    MPI_Recv(message, 1, MPI_INT, source, HEARTBEAT_TAG, MPI_COMM_WORLD, &status);
    return message[0];
}

void p2p_send_termination_vote(int32_t target, int32_t local_count) {
    int32_t message[1];
    message[0] = local_count;
    MPI_Send(message, 1, MPI_INT, target, TERMINATION_VOTE_TAG, MPI_COMM_WORLD);
}

int32_t p2p_receive_termination_vote(int32_t source) {
    MPI_Status status;
    int32_t message[1];
    MPI_Recv(message, 1, MPI_INT, source, TERMINATION_VOTE_TAG, MPI_COMM_WORLD, &status);
    return message[0];
}

void p2p_send_termination_confirm(int32_t target) {
    int32_t message[1];
    message[0] = 1;
    MPI_Send(message, 1, MPI_INT, target, TERMINATION_CONFIRM_TAG, MPI_COMM_WORLD);
}

void p2p_receive_termination_confirm(int32_t source) {
    MPI_Status status;
    int32_t message[1];
    MPI_Recv(message, 1, MPI_INT, source, TERMINATION_CONFIRM_TAG, MPI_COMM_WORLD, &status);
}
