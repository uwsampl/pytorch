#pragma once

#include <vector>
#include <queue>

template<typename T>
struct PairCompare
{
    bool operator()(const std::pair<double, T>& left, const std::pair<double, T>& right) const
    {
        return left.first > right.first;
    }
};

template <typename T>
struct GdsHeap {
    typedef std::pair<double, T> node_type;
    std::priority_queue<node_type, std::vector<node_type>, PairCompare<T>> queue;
    double L = 0.0;

    bool empty() const {
        return queue.empty();
    }

    size_t size() const {
        return queue.size();
    }

    void push(const double cost, const T& value) {
        queue.push(std::make_pair(cost + L, value));
    }

    std::pair<double, const T&> top() const {
        return std::make_pair(queue.top().first - L, queue.top().second);
    }

    void pop() {
        L += top().first;
        queue.pop();
    }
};
