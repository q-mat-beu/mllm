#pragma once

#include "layer.h"
#include <vector>

class Model {
private:
    std::vector<Layer> layers;
public:
    Model() = default;
};
