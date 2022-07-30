from __future__ import print_function
__autor__ = 'hwpoison'


"""
	Implementation of the neuronal model of Warren McCulloch and Walter Pitts.
					"Linear Threshould Unit"
"""


class MPNeuron(object):
    def __init__(self):
        super(MPNeuron).__init__()
        self.threshold = 0

    def activation(self, inputs_sum):
        # if the sum of the entries exceeds the threshold return 1
        return 1 if inputs_sum >= self.threshold else 0

    def input(self, inputs, inhibitory=[]):
        # detects if there is an inhibitory input activated
        for inhibited_input in inhibitory:
            if(inputs[inhibited_input]):
                return 0

        # aggregation, type sum
        return self.activation(sum(inputs))


if __name__ == '__main__':
    # instance
    mp = MPNeuron()

    def logic_gates():
        logic_gate = [
            [1, 0],
            [0, 1],
            [0, 0],
            [1, 1]
        ]
        mp.threshold = 2
        print("---- AND GATE ----")
        for n_input in logic_gate:
            print(n_input, mp.input(n_input))

        mp.threshold = 1
        print("---- OR GATE ----")
        for n_input in logic_gate:
            print(n_input, mp.input(n_input))

    def go_cine():
        print("--- CINEMA PROBLEM ----")
        # i want go to cinema, i love robots movies
        # but i hate when is a comedy robot genre
        # and i can't go if is rain :(
        # format: [HAVE ROBOTS - IS COMEDY - IS RAIN]
        # IS COMEDY and IS RAIN inputs are inhibitory, if is 1 , return 0

        situations = [
            # Automata
            [1, 0, 0],  # expect 1

            # Blade runner
            [1, 0, 0],  # expected 1

            # Hot bot
            [1, 1, 0],  # expect 0

            # I am mother
            [1, 0, 1],  # expect 0, is rain x.x

            # Pulp fiction
            [0, 0, 0],  # expect 0, wtf xD
        ]
        mp.threshold = 1  # only one genre
        for n_input in situations:
            print(n_input, mp.input(n_input, [1, 2]))
    logic_gates()

    go_cine()
