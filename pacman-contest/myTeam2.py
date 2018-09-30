# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint

def createTeam(firstIndex, secondIndex, isRed,
               first = 'MyAgent', second = 'MyAgent'):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(firstIndex), eval(second)(secondIndex)]



class MyAgent(CaptureAgent):

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)
        # Always initiated as offense agent
        self.mode = "offense"
        self.capsule_time = 0

    ##################
    # Mode & Actions #
    ##################
    def chooseAction(self, gameState):
        """
        1. Decide if agent should take offensive or defensive action
        2. Picks among the actions with the highest Q(s,a).
        """
        start = time.time()
        # update capsule time
        if self.getPreviousObservation()!= None:
            if len(self.getCapsules(self.getPreviousObservation())) - len(self.getCapsules(gameState)):
                self.capsule_time = 38
            else:
                self.capsule_time = max(0, self.capsule_time - 1)

        # Choose Mode
        myTeam = self.getTeam(gameState)
        oppoTeam = self.getOpponents(gameState)
        partner_index = [a for a in myTeam if a != self.index][0]
        partner = gameState.getAgentState(partner_index)
        partnerPos = partner.getPosition()
        myPos = gameState.getAgentPosition(self.index)
        score = self.getScore(gameState)
        upper_bound = 13 # Change this
        lower_bound = -1
        pac = [1 for i in oppoTeam if gameState.getAgentState(i).isPacman]
        if not pac:
            num_pac = 0
        else:
            num_pac = sum(pac)

        ## Choose mode based on utility
        # FIXME: if we become white, we go offense
        if num_pac == 0 or score <= lower_bound: # All opponents are ghosts
            self.mode = "offense"
        elif num_pac == 2 or score >= upper_bound:
            self.mode = "defense"
        else:
            if myPos[0] > partnerPos[0]:
                self.mode = "offense"
            else:
                self.mode = "defense"

        # Choose Action
        actions = gameState.getLegalActions(self.index)
        is_offense_minimax = False # if we need to use min-max
        is_defense_minimax = False
        for oppo in oppoTeam:
            if gameState.getAgentPosition(oppo) is not None and not gameState.getAgentState(oppo).isPacman and not self.capsule_time:
                is_offense_minimax = True
            if gameState.getAgentPosition(oppo) is not None and gameState.getAgentState(oppo).isPacman and not self.capsule_time:
                is_defense_minimax = True

        ## Min-Max needed
        if is_offense_minimax and self.mode == "offense":
            bestScore, alpha = -sys.maxint, -sys.maxint
            action = None
            for a in actions:
                score = self.offenseValue(gameState.generateSuccessor(self.index, a), self.index,
                                          alpha, sys.maxint, 4, gameState.getNumAgents())
                if score >= bestScore:
                    bestScore = score
                    action = a
                if score > alpha:
                    alpha = score
            # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)
            return action


        ## Min-Max not needed
        else:
            values = [self.evaluate(gameState, a) for a in actions]
            maxValue = max(values)
            bestActions = [a for a, v in zip(actions, values) if v == maxValue]

            if self.mode == "offense":
                foodLeft = len(self.getFood(gameState).asList())

                if foodLeft <= 2:
                    bestDist, bestAction = 9999, None
                    for action in actions:
                        successor = self.getSuccessor(gameState, action)
                        pos2 = successor.getAgentPosition(self.index)
                        dist = self.getMazeDistance(self.start, pos2)
                        if dist < bestDist:
                            bestAction = action
                            bestDist = dist
                    return bestAction
            action = random.choice(bestActions)
            return action

    #####################
    # Feature & Weights #
    #####################
    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        myAgent = successor.getAgentState(self.index)
        myPos = myAgent.getPosition()

        if self.mode == "offense":
            # Successor Score: How much score higher than the other team
            features['successorScore'] = self.getScore(successor)

            # Food To Eat: how many food we need to eat to beat the other team
            foodList = self.getFood(successor).asList()
            features['foodToEat'] = len(self.getFoodYouAreDefending(successor).asList()) - len(foodList)

            # Min Score Dist: Minimum
            min_dist_to_score = min(
                [self.getMazeDistance(myPos, (16, float(i))) for i in range(0, 16) if not successor.hasWall(16, i)])
            if self.red:
                min_dist_to_score = min(
                    [self.getMazeDistance(myPos, (15, float(i))) for i in range(0, 16) if not successor.hasWall(15, i)])
            features['minScoreDist'] = -min_dist_to_score

            # Distance to Food: distance to the nearest food
            minDistance = 0
            if len(foodList) > 0:
                minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = -minDistance


            # Distance to Ghost: how far are we from the closest ghost
            min_dist_to_ghost = 5
            for oppo in self.getOpponents(successor):
                if not successor.getAgentState(oppo).isPacman:
                    if successor.getAgentPosition(oppo) is None:
                        continue
                    dist = self.getMazeDistance(myPos, successor.getAgentPosition(oppo))
                    if dist < min_dist_to_ghost or not min_dist_to_ghost:
                        min_dist_to_ghost = dist
            if self.capsule_time > 0: # Opponent is white ghost
                min_dist_to_ghost = 5
            # if myPos == successor.getInitialAgentPosition(self.index):
            #     min_dist_to_ghost = -5 # We got eaten
            features['distanceToGhost'] = min_dist_to_ghost


            # Distance to Capsules
            dist_to_capsules = 0
            capPos = self.getCapsules(successor)
            if capPos:
                capPos = capPos[0]
                dist_to_capsules = self.getMazeDistance(myPos, capPos)
            features['distanceToCapsules'] = -dist_to_capsules

            # ifDeadEnd: 3 walls around
            deadEnd = 0
            if len(successor.getLegalActions(self.index)) == 1:
                deadEnd = -1
            elif len(successor.getLegalActions(self.index)) == 2:
                deadEnd = -0.5
            features['isDeadEnd'] = deadEnd

            # Distance to Pacman: Attack opponent pacman when we haven't get over yet
            dist_to_pacman = 0
            if not myAgent.isPacman:
                for oppo in self.getOpponents(successor):
                    if successor.getAgentState(oppo).isPacman \
                            and successor.getAgentPosition(oppo) is not None:
                        dist = self.getMazeDistance(myPos, successor.getAgentPosition(oppo))
                        if dist_to_pacman and dist < dist_to_pacman:
                            dist_to_pacman = dist
            features['distanceToPacman'] = -dist_to_pacman

            # Distance to partner
            dist_to_partner = 0
            myTeam = self.getTeam(gameState)
            partner_index = [a for a in myTeam if a != self.index][0]
            partner = gameState.getAgentState(partner_index)
            partnerPos = partner.getPosition()
            dist_to_partner = self.getMazeDistance(myPos, partnerPos)
            features['distanceToPartner'] = dist_to_partner**2

        # Defense
        else:
            # num Invaders: distance to invaders we can see
            enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
            invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
            features['numInvaders'] = -len(invaders)
            if len(invaders) > 0:
                dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
                features['invaderDistance'] = -min(dists)

            if action == Directions.STOP: features['stop'] = -1
            rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
            if action == rev: features['reverse'] = -1

        return features

    def getWeights(self, gameState, action):
        if self.mode == "offense":
            # How many food on us we haven't delivered back
            food_bearing = 20 - len(self.getFood(gameState).asList()) - self.getScore(gameState)
            return {'successorScore': 10000,
                    'distanceToFood': 1,
                    'foodToEat': 1000,
                    'minScoreDist': 2 * food_bearing,
                    'distanceToGhost': 500 * (food_bearing + 1),
                    'distanceToCapsules': 10,
                    'isDeadEnd': 10,
                    'distanceToPacman': 10,
                    'distanceToPartner': 0}
        else:
            return {'numInvaders': 1000,
                    'invaderDistance': 10,
                    'stop': 100,
                    'reverse': 2}

    ####################
    # Helper Functions #
    ####################
    def evaluate_minimax(self, gameState):
        actions = gameState.getLegalActions(self.index)
        values = [self.evaluate(gameState, a) for a in actions]
        maxValue = max(values)
        return maxValue

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def cutoffTest(self, gameState, depth):
        if depth == 0 or gameState.isOver():
            return True
        return False

    #####################
    # Offensive MiniMax #
    #####################
    def offenseValue(self, gameState, agentIndex, alpha, beta, depth, numAgents):
        if self.cutoffTest(gameState, depth):
            return self.evaluate_minimax(gameState)

        if agentIndex == self.index:
            return self.offenseMax(gameState, agentIndex, alpha, beta, depth, numAgents)
        elif gameState.isOnRedTeam(agentIndex) != gameState.isOnRedTeam(self.index) \
                and gameState.getAgentPosition(agentIndex) != None \
                and not gameState.getAgentState(agentIndex).isPacman:
            return self.offenseMin(gameState, agentIndex, alpha, beta, depth, numAgents)
        else:
            return self.offenseValue(gameState, (agentIndex + 1) % numAgents, alpha, beta, depth, numAgents)

    def offenseMax(self, gameState, agentIndex, alpha, beta, depth, numAgents):
        v = -sys.maxint
        for a in gameState.getLegalActions(agentIndex):
            s = gameState.generateSuccessor(agentIndex, a)
            v = max(v, self.offenseValue(s, ((agentIndex + 1) % numAgents), alpha, beta, depth - 1, numAgents))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def offenseMin(self, gameState, agentIndex, alpha, beta, depth, numAgents):
        v = sys.maxint
        past_dist = self.getMazeDistance(gameState.getAgentPosition(self.index), gameState.getAgentPosition(agentIndex))
        for a in gameState.getLegalActions(agentIndex):
            s = gameState.generateSuccessor(agentIndex, a)
            if self.getMazeDistance(s.getAgentPosition(self.index), gameState.getAgentPosition(self.index)) >= 2 and s.getAgentPosition(self.index) == s.getInitialAgentPosition(self.index):
                return -sys.maxint
            if self.getMazeDistance(s.getAgentPosition(self.index),
                                    s.getAgentPosition(agentIndex)) < past_dist or self.getMazeDistance(s.getAgentPosition(self.index),
                                                                                                        s.getAgentPosition(agentIndex)) <= 2:
                v = min(v, self.offenseValue(s, ((agentIndex + 1) % numAgents), alpha, beta, depth - 1, numAgents))
                if v <= alpha:
                    return v
                beta = min(beta, v)
        return v





