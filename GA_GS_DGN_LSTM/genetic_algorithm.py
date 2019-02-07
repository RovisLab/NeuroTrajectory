"""Entry point to evolving the neural network. Start here."""
from __future__ import print_function
from evolver import Evolver
from tqdm import tqdm
from load_data import *
import csv
import datetime
import time
import logging


class GeneticAlgorithm:
    def __init__(self, path, params, model, data_set):
        self.path = path + '/genetic_algorithm'
        self.data_set = data_set
        self.params = params
        self.model = model
        self.population = 20
        self.generations = 8

        self.create_dirs()

    def create_dirs(self):
        os.makedirs(self.path)
        os.makedirs(self.path + '/models')
        os.makedirs(self.path + '/plots')
        os.makedirs(self.path + '/confusion_matrix')
        os.makedirs(self.path + '/conf_matrix_csv')
        os.makedirs(self.path + '/conf_matrix_details')

    def run(self):
        print("***Evolving for %d generations with population size = %d***" % (self.generations, self.population))
        self.generate()

    def train_genomes(self, genomes, writer):
        logging.info("***train_networks(networks, dataset)***")
        pbar = tqdm(total=len(genomes))

        for genome in genomes:
            genome.train(self.model, self.data_set, self.path)

            parameters = list()
            params_csv = list()

            for p in self.params:
                parameters.append(genome.geneparam[p])
                params_csv.append(str(genome.geneparam[p]))

            params_csv.append(genome.accuracy)
            row = params_csv
            writer.writerow(row)
            pbar.update(1)

        pbar.close()

    def generate(self):
        logging.info("***generate(generations, population, all_possible_genes, dataset)***")
        t_start = datetime.datetime.now()
        t = time.time()

        evolver = Evolver(self.params)
        genomes = evolver.create_population(self.population)

        ofile = open(self.path + '/result.csv', "w")
        writer = csv.writer(ofile, delimiter=',')

        table_head = list()
        for p in self.params:
             table_head.append(str(p))

        table_head.append("accuracy")
        row = table_head
        writer.writerow(row)

        # Evolve the generation.
        for i in range(self.generations):
            logging.info("***Now in generation %d of %d***" % (i + 1, self.generations))
            self.print_genomes(genomes)

            # Train and get accuracy for networks/genomes.
            self.train_genomes(genomes, writer)

            # Get the average accuracy for this generation.
            average_accuracy = self.get_average_accuracy(genomes)

            # Print out the average accuracy each generation.
            logging.info("Generation average: %.2f%%" % (average_accuracy * 100))
            logging.info('-'*80)

            # Evolve, except on the last iteration.
            if i != self.generations - 1:
                genomes = evolver.evolve(genomes)

        # Sort our final population according to performance.
        genomes = sorted(genomes, key=lambda x: x.accuracy, reverse=True)

        # Print out the top 5 networks/genomes.
        self.print_genomes(genomes[:5])

        ofile.close()
        total = time.time() - t
        t_stop = datetime.datetime.now()
        file = open(self.path + '/total_time.txt', 'w')
        file.write('Start : ' + str(t_start) + '\n')
        file.write('Stop : ' + str(t_stop) + '\n')
        file.write('Total : ' + str(total) + '\n')
        file.close()

    @staticmethod
    def get_average_accuracy(genomes):
        total_accuracy = 0
        for genome in genomes:
            total_accuracy += genome.accuracy

        return total_accuracy / len(genomes)

    @staticmethod
    def print_genomes(genomes):
        logging.info('-'*80)
        for genome in genomes:
            genome.print_genome()
