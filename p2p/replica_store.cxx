#include "replica_store.hxx"

#include <algorithm>
#include <cmath>
#include <limits>

#include "common/log.hxx"

ReplicaStore::ReplicaStore(int32_t _max_per_island) : max_per_island(_max_per_island) {}

ReplicaStore::~ReplicaStore() {
    for (auto& pair : replicas) {
        for (RNN_Genome* genome : pair.second) {
            delete genome;
        }
    }
    replicas.clear();
}

void ReplicaStore::store_replica(int32_t global_island_id, RNN_Genome* genome) {
    RNN_Genome* copy = genome->copy();

    auto& island_replicas = replicas[global_island_id];

    // Insert in sorted order (best fitness first, lower is better)
    auto it = island_replicas.begin();
    while (it != island_replicas.end() && (*it)->get_fitness() < copy->get_fitness()) {
        ++it;
    }
    island_replicas.insert(it, copy);

    // Evict worst if over capacity
    while ((int32_t)island_replicas.size() > max_per_island) {
        RNN_Genome* worst = island_replicas.back();
        island_replicas.pop_back();
        delete worst;
    }

    Log::debug("ReplicaStore: stored replica for island %d, now have %d replicas\n",
               global_island_id, (int32_t)island_replicas.size());
}

vector<RNN_Genome*> ReplicaStore::get_replicas(int32_t global_island_id) const {
    auto it = replicas.find(global_island_id);
    if (it == replicas.end()) {
        return {};
    }
    return it->second;
}

void ReplicaStore::clear_island(int32_t global_island_id) {
    auto it = replicas.find(global_island_id);
    if (it != replicas.end()) {
        for (RNN_Genome* genome : it->second) {
            delete genome;
        }
        it->second.clear();
        replicas.erase(it);
    }
}

int32_t ReplicaStore::get_replica_count(int32_t global_island_id) const {
    auto it = replicas.find(global_island_id);
    if (it == replicas.end()) {
        return 0;
    }
    return (int32_t)it->second.size();
}
