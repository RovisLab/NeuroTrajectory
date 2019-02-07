"""The genome to be evolved."""
from training_history_plot import TrainingHistoryPlot
from keras.callbacks import EarlyStopping
import random
import logging
import hashlib
import copy
import numpy as np
from keras import backend as K


class Genome():
    """
    Represents one genome and all relevant utility functions (add, mutate, etc.).
    """
    def __init__(self, all_possible_genes=None, geneparam={}, u_ID=0, mom_ID=0, dad_ID=0, gen=0):
        """Initialize a genome.

        Args:
            all_possible_genes (dict): Parameters for the genome
        """
        self.accuracy = 0.0
        self.all_possible_genes = all_possible_genes

        # (dict): represents actual genome parameters
        self.geneparam = geneparam
        self.u_ID = u_ID
        self.parents = [mom_ID, dad_ID]
        self.generation = gen
        
        #hash only makes sense when we have specified the genes
        if not geneparam:
            self.hash = 0
        else:
            self.update_hash()
        
    def update_hash(self):
        """
        Refesh each genome's unique hash - needs to run after any genome changes.
        """
        genh = ''
        for p in self.all_possible_genes:
            genh += str(self.geneparam[p])

        self.hash = hashlib.md5(genh.encode("UTF-8")).hexdigest()
        self.accuracy = 0.0
            
    def set_genes_random(self):
        """Create a random genome."""
        self.parents = [0, 0]
        for key in self.all_possible_genes:
            self.geneparam[key] = random.choice(self.all_possible_genes[key])
                
        self.update_hash()
        
    def mutate_one_gene(self):
        """Randomly mutate one gene in the genome.

        Args:
            network (dict): The genome parameters to mutate

        Returns:
            (Genome): A randomly mutated genome object

        """
        # Which gene shall we mutate? Choose one of N possible keys/genes.
        gene_to_mutate = random.choice(list(self.all_possible_genes.keys()))

        # And then let's mutate one of the genes.
        # Make sure that this actually creates mutation
        current_value = self.geneparam[gene_to_mutate]
        possible_choices = copy.deepcopy(self.all_possible_genes[gene_to_mutate])
        possible_choices.remove(current_value)
        self.geneparam[gene_to_mutate] = random.choice( possible_choices )
        self.update_hash()
    
    def set_generation(self, generation):
        """needed when a genome is passed on from one generation to the next.
        the id stays the same, but the generation is increased"""
        self.generation = generation

    def set_genes_to(self, geneparam, mom_ID, dad_ID):
        """Set genome properties.
        this is used when breeding kids

        Args:
            genome (dict): The genome parameters
        IMPROVE
        """
        self.parents = [mom_ID, dad_ID]
        self.geneparam = geneparam
        self.update_hash()

    def train_and_score(self, model_train, dataset, path):
        logging.info("Getting training samples")
        logging.info("Compling Keras model")

        batch_size = self.geneparam['batch_size']
        epochs = self.geneparam['epochs']

        parameters = list()

        for p in self.all_possible_genes:
            if p is 'batch_size':
                continue
            elif p is 'epochs':
                continue
            else:
                parameters.append(self.geneparam[p])

        print(parameters)
        input_shape = np.shape(dataset.X_train)
        model = model_train(input_shape, parameters)

        parameters.append(self.geneparam['batch_size'])
        parameters.append(self.geneparam['epochs'])
        # Helper: Early stopping.
        early_stopper = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=5, verbose=0, mode='auto')
        history = TrainingHistoryPlot(path, dataset, parameters)
        model.fit(dataset.X_train, dataset.Y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(dataset.X_valid, dataset.Y_valid),
                  callbacks=[early_stopper, history])

        score = model.evaluate(dataset.X_valid, dataset.Y_valid, verbose=0)
        K.clear_session()
        # we do not care about keeping any of this in memory -
        # we just need to know the final scores and the architecture

        # 1 is accuracy. 0 is loss.
        return score[1]

    def train(self, model, trainingset, path):
        #don't bother retraining ones we already trained
        if self.accuracy == 0.0:
            self.accuracy = self.train_and_score(model, trainingset, path)

    def print_genome(self):
        """Print out a genome."""
        self.print_geneparam()
        logging.info("Acc: %.2f%%" % (self.accuracy * 100))
        logging.info("UniID: %d" % self.u_ID)
        logging.info("Mom and Dad: %d %d" % (self.parents[0], self.parents[1]))
        logging.info("Gen: %d" % self.generation)
        logging.info("Hash: %s" % self.hash)

    def print_genome_ma(self):
        """Print out a genome."""
        self.print_geneparam()
        logging.info("Acc: %.2f%% UniID: %d Mom and Dad: %d %d Gen: %d" % (self.accuracy * 100, self.u_ID,
                                                                           self.parents[0], self.parents[1],
                                                                           self.generation))
        logging.info("Hash: %s" % self.hash)

    # print nb_neurons as single list
    def print_geneparam(self):
        g = self.geneparam.copy()
        logging.info(g)
    
    # convert nb_neurons_i at each layer to a single list
    def nb_neurons(self):
      nb_neurons = [None] * 6
      for i in range(0,6):
        nb_neurons[i] = self.geneparam['nb_neurons_' + str(i+1)]

      return nb_neurons
