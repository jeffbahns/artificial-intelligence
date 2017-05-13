from numpy import array, fliplr, flipud, linalg, isnan, subtract, dot, random, multiply
from random import choice, shuffle
import mnist_data as md

def density(m):
    # 1's / rows*columns
    ones = 0
    for row in m:
        for item in row:
            if item == 1:
                ones += 1
    total_cells = len(m) * len(m[0])
    #print("ones : {} , total : {}, density : {}".format(ones,total_cells, (1.0*ones)/total_cells))
    return (1.0*ones) / total_cells

def symmetry(m, _horizontal):
    px = array(m)
    flip = fliplr(px) if _horizontal else flipud(px)
    x = linalg.norm(subtract(flip, px))
    y = x / density(px)
    val = 1 - (x / y)
    return 1 if isnan(val) else val
    
def intersections(m, _max, _horizontal):
    intersections = 0
    upper_i = len(m)
    upper_j = len(m[0])
    for i in range(0, upper_i if _horizontal else upper_j):
        _last = 0
        _intersections = 0
        for j in range(0, upper_j if _horizontal else upper_i):
            if _last != (m[i][j] if _horizontal else m[j][i]):
                _intersections += 1
                _last = m[i][j] if _horizontal else m[j][i]
        intersections = max(intersections, _intersections) if _max else max(2,min(intersections, _intersections))
        #intersections = max(intersections, _intersections) if _max else min(intersections, _intersections)
    return intersections

def generate_features(m):
    densityf = density(m) # 1 density
    h_symmetryf = symmetry(m, True) # 2 horizontal symmetry
    v_symmetryf = symmetry(m, False) # 3 vertical symmetry
    h_max_intersections = intersections(m, True, True) # 4 horizontal max  MINS always == 0 ASK ravi
    h_min_intersections = intersections(m, False, True) # 5 horizontal min
    v_max_intersections = intersections(m, True, False) # 6 vertical max
    v_min_intersections = intersections(m, False, False) # 7 vertical min
    #return [densityf, [h_symmetryf, v_symmetryf], [h_max_intersections, h_min_intersections, v_max_intersections, v_max_intersections], 1]
    return [densityf, h_symmetryf, v_symmetryf, h_max_intersections, h_min_intersections, v_max_intersections, v_min_intersections,1]
    #return [densityf, h_symmetryf, v_symmetryf]

def pocket_perceptron(training_data,unknown_data):
    num_features = len(training_data[0][0])
    w = random.rand(num_features) # weights <w1, w2, ... , w7>
    #print("w:",w)
    eta = .2
    n = 100
    correct = 0
    unit_step = lambda x: 1 if x < 0 else 5
    for i in range(0, n):
        td_copy = training_data[:]
        shuffle(td_copy)
        for x, expected in td_copy:
            result = dot(w, x)
            error = expected - unit_step(result)
            w += multiply(eta*error, x)
        #x, expected = choice(training_data)
        #result = dot(w, x)
        #error = expected - unit_step(result)
        #w += multiply(eta*error, x)
        #print(w)
    for x, expected in unknown_data:
        result = dot(x, w)
        if unit_step(result) == expected:
            correct += 1
        #print("{}: {} -> {} == {}".format(x[:num_features], result, unit_step(result), expected))
    print("Accuracy :: {} %".format((correct / len(unknown_data)) * 100))
    #print("w:",w)

def main():
    one100 = md.one100
    one900 = md.one900
    five100 = md.five100
    five900 = md.five900

    training_data = []
    unknown_data = []
    # training data
    for one in one900:
        training_data.append((generate_features(one), 1))
    for five in five900:
        training_data.append((generate_features(five), 5))

    # unknown unknown
    for one in one100:
        unknown_data.append((generate_features(one), 1))
    for five in five100:
        unknown_data.append((generate_features(five), 5))

    pocket_perceptron(training_data, unknown_data)
    return 0

main()
