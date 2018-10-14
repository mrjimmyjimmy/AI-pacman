from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util
from game import Directions
import game
from util import nearestPoint
import copy
from math import sqrt, log
import random, time
import baselineTeam


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveReflexAgent', second='DefensiveReflexAgent'):
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that chooses score-maximizing actions
    """

    def __init__(self, gameState):
        CaptureAgent.__init__(self, gameState)
        self.powerTimer = 0

    def legalPosition(self, previousPosition, nextPosition):

        x, y = previousPosition
        m, n = nextPosition
        if (m == x and (n == y + 1 or n == y - 1)) or (n == y and (m == x + 1 or m == x - 1)):
            return True
        else:
            return False

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        '''
        We set all the inital weights to 0, and using Q learning to update the weight
        '''
        # self.weights = {'score': 0, 'DisToNearestFood': 0, 'disToGhost':0,
        #                 'disToCapsule': 0, 'dots': 0,
        #                 'disToBoundary': 0,'deadends':0}
        self.weights = {'score': 1.78261354182, 'DisToNearestFood': -4.91094492098, 'disToGhost': 8.17572535548,
                        'disToCapsule': -1.36111562824, 'dots': -0.877933155097,
                        'disToBoundary': -2.94156916302, 'deadends': -10}

        self.distancer.getMazeDistances()

        # ----------- DEADEND PROCESSING
        self.deadEnds = getDeadEnds(gameState, self.red)

        if self.red:
            cX = (gameState.data.layout.width - 2) / 2
            self.delta = 1
        else:
            cX = ((gameState.data.layout.width - 2) / 2) + 1
            self.delta = -1

        # get red team's boundary and blue team's boundary
        self.boundary = []
        self.boundaryBlue = []
        for i in range(1, gameState.data.layout.height - 1):
            if not gameState.hasWall(cX, i):
                self.boundary.append((cX, i))
            if not gameState.hasWall(cX + self.delta, i):
                self.boundaryBlue.append((cX + self.delta, i))

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

    def evaluate(self, gameState, action, agentType):
        '''
        Use different function to evaluate based on the agent type
        '''
        if agentType == 'defence':
            return self.evl3(gameState,action)

        if agentType == 'offence':
            return self.evl2(gameState, action)

        if agentType == 'move':
            features = self.getFeatureMove(gameState,action)
            weights = self.getWeightMove()
            return features * weights


    # ####################
    # # agent type: move #
    # ####################
    def getFeatureMove(self, gameState, action):
        # initial features
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        # get the position
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()
        point = self.enemyArrayPoint(gameState)
        features['disToEnemyArrive'] = self.getMazeDistance(point,myPos)
        return features

    def getWeightMove(self):
        return {'disToEnemyArrive': -1}




    # #######################
    # # agent type: offence #
    # #######################
    def getFeaturesOffense(self, previous, action):
        features = util.Counter()
        gameState = self.getSuccessor(previous, action)
        previousPosition = previous.getAgentState(self.index).getPosition()
        agentPosition = gameState.getAgentState(self.index).getPosition()

        # ---------------------feature: pacman be eaten ------------------
        if not self.legalPosition(previousPosition, agentPosition):  # back to born place
            features['disToGhost'] = -100
            return features

        # ---------------------feature 1: score----------------
        features['score'] = self.getScore(gameState)
        # ---------------------feature 2: distance to closest food----------------
        x, y = gameState.getAgentState(self.index).getPosition()
        oldFood = self.getFood(previous)
        if oldFood[int(x)][int(y)]:
            features['DisToNearestFood'] = 0
        else:
            food = self.getFood(gameState).asList()
            if len(food) <= 2:
                features['DisToNearestFood'] = 0
            if len(food) > 2:
                dis = []
                for f in food:
                    dis.append(self.getMazeDistance(agentPosition, f))
                minDis = min(dis)
                features['DisToNearestFood'] = minDis

        # ---------------------feature 3: dis to closest ghost----------------
        enemies = []
        for e in self.getOpponents(gameState):
            enemyState = gameState.getAgentState(e)
            if not enemyState.isPacman and not enemyState.getPosition() is None and not enemyState.scaredTimer > 5:
                enemies.append(enemyState)
        if len(enemies) > 0:
            toEnemies = []
            for e in enemies:
                enemyPos = e.getPosition()
                toEnemies.append(self.getMazeDistance(agentPosition, enemyPos))
            dis = min(toEnemies)
            features['disToGhost'] = dis
            if dis >= 12:
                features['disToGhost'] = 12
        else:
            features['disToGhost'] = 12

        # ---------------------feature 4: dis to closest capsule----------------
        oldCapsule = self.getCapsules(previous)
        capsule = self.getCapsules(gameState)
        if (int(x), int(y)) in oldCapsule:
            features['disToCapsule'] = 0  # eat capsule from previous state via action to the gamestate
        else:
            if len(capsule) == 0:
                features['disToCapsule'] = 0
            else:
                dis = []
                for c in capsule:
                    dis.append(self.getMazeDistance(agentPosition, c))
                features['disToCapsule'] = min(dis)

        # ---------------------feature 5: carrying----------------
        features['dots'] = gameState.getAgentState(self.index).numCarrying
        features['oldDots'] = previous.getAgentState(self.index).numCarrying
        # ---------------------feature 6: dis to boundary----------------
        disToBoundary = 99999
        for a in range(len(self.boundary)):
            disToBoundary = min(disToBoundary, self.getMazeDistance(agentPosition, self.boundary[a]))
        features['disToBoundary'] = disToBoundary

        # ----------------------feature 7: deadends-----------
        features['deadends'] = 0
        if self.deadEnds.has_key((previous.getAgentState(self.index).getPosition(), action)) and (
                features['disToGhost'] < 12 or self.disToNearestGhost(previous) < 6) and self.deadEnds[
            (previous.getAgentState(self.index).getPosition(), action)] * 2 >= features['disToGhost'] - 1 > 0 and not (self.isTowardsCapsule(previous,action)):
            features['deadends'] = 100

        # ------------------------feature 8 : timeLeft ---------
        features['timeLeft'] = previous.data.timeleft

        # ---------------------feature 9: strong---------------
        # used when pacman eat an capsules
        if agentPosition in self.getCapsules(previous):
            self.powerTimer = 100
        features['strong'] = self.powerTimer
        # If powered, reduce power timer each itteration
        if self.powerTimer > 0:
            self.powerTimer -= 1

        return features

    def getWeightOffence(self, gameState, action):
        return self.weights



    # #######################
    # # agent type: defence #
    # #######################

    def getFeaturesDefence(self, gameState, action):
        # initial features
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        # get the position
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()
        nextActions = successor.getLegalActions(self.index)

        #------------feature1: Computes whether we're on defense (1) or offense (0)--------------
        features['onDefense'] = 1
        if myState.isPacman: features['onDefense'] = 0

        # ------------feature2: enemy pacman distance -----------
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)


        # ------------ features3: aviod enemy when we are sacred or we are pacman and we want to defence--------------------
        if myState.scaredTimer > 0 and 0 <= features['invaderDistance'] < 4 > 0:
            features['dangerDistance'] = features['invaderDistance']
            features['invaderDistance'] = 0
            features['nextdead'] = len(nextActions)
        # aviod go to deadend
        if myState.isPacman or myState.scaredTimer > 0:
            features['deadends'] = 0
            if self.deadEnds.has_key((gameState.getAgentState(self.index).getPosition(), action)) and \
                    (features['disToGhost'] < 12 or self.disToNearestGhost(gameState) < 6)and self.deadEnds[
                (gameState.getAgentState(self.index).getPosition(), action)] * 2 >= features['disToGhost'] - 1 > 0:
                features['deadends'] = 100


        # ------------ features4: distance to bounary --------------------
        # there is no enemy in our filed, hang out at boundary
        disToBoundary = 99999
        for e in enemies:
            if not e.isPacman:
                for a in range(len(self.boundary)):
                    disToBoundary = min(disToBoundary, self.getMazeDistance(myPos, self.boundary[a]))
                features['disToBoundary'] = disToBoundary


        # ------------ features5: track where we lost food --------------------
        # get the distance where the food lost
        if not self.foodLost == []:
            minDistance = self.getMazeDistance(myPos, self.foodLost[0])
            features['lostFoodDistance'] = minDistance
        if not self.foodLostPosition() == None:
            food = self.foodLostPosition()[0]
            self.foodLost = []
            self.foodLost.append(food)
            minDistance = self.getMazeDistance(myPos, self.foodLost[0])
            features['lostFoodDistance'] = minDistance
        # if there is no enemy, we wont go lost food position
        enemiesState = []
        for e in enemies:
            if e.isPacman:
                enemiesState.append(e)
        if len(enemiesState) == 0:
            features['lostFoodDistance'] = 0

        return features

    def getWeightsDefence(self, gameState, action):
        return {'numInvaders': -10, 'onDefense': 1, 'invaderDistance': -10000, 'deadends': -10,
                'lostFoodDistance': -1000, 'disToBoundary': -20, 'dangerDistance': 10000, 'nextdead': 1}


    def evl3(self, gameState, action):
        '''
        adjust our features and weights when we facing different situation;
        used for defence agent
        '''
        features = self.getFeaturesDefence(gameState, action)
        weights = self.getWeightsDefence(gameState, action)
        newWeights = copy.deepcopy(weights)
        self.offenceMode = 'normal'
        successor = self.getSuccessor(gameState, action)

        # get the position
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()
        previousPosition = gameState.getAgentState(self.index).getPosition()

        if gameState.getAgentState(self.index).isPacman:
            if not self.legalPosition(previousPosition, myPos):  # back to born place
                features['disToGhost'] = -100
                return features
            if features['disToGhost'] == 1:
                if (self.getSuccessor(gameState, action).getAgentState(self.index).isPacman):
                    newWeights['disToGhost'] = -100

        return features * newWeights



    def evl2(self, gameState, action):
        '''
        adjust our features and weights when we facing different situation;
        used for offence agent
        '''
        features = self.getFeaturesOffense(gameState, action)
        weights = self.getWeightOffence(gameState, action)
        newWeights = copy.deepcopy(weights)
        self.offenceMode = 'normal'

        # ---------situation 1, pacman is currying less than 8, try to get more
        if features['oldDots'] <= 7:
            newWeights = {'score': 20.78261354182, 'DisToNearestFood': -7.91094492098, 'disToGhost': 8.17572535548,
                          'disToCapsule': -4.36111562824, 'dots': -0.877933155097,
                          'disToBoundary': -2.94156916302, 'deadends': -10, }
            if len(self.getCapsules(gameState)) == 0:
                newWeights['DisToNearestFood'] = -13.21094492098

            if features['disToGhost'] == 1:
                if (self.getSuccessor(gameState, action).getAgentState(self.index).isPacman):
                    newWeights['disToGhost'] = -100

        # ---------situaion 2, pacman is currying more than 9, try to go home
        if features['oldDots'] >= 8 and self.disToNearestGhost(gameState) < 6:
            newWeights = {'score': 20.78261354182, 'DisToNearestFood': -2.91094492098, 'disToGhost': 8.17572535548,
                          'disToCapsule': -1.36111562824, 'dots': -0.877933155097,
                          'disToBoundary': -6.94156916302, 'deadends': -10}
            if features['disToGhost'] == 1:
                if (self.getSuccessor(gameState, action).getAgentState(self.index).isPacman):
                    newWeights['disToGhost'] = -100

        if features['timeLeft'] < 200 and gameState.getAgentState(self.index).numCarrying != 0:
            newWeights['disToBoundary'] = -15

        # --------situation 3, pacman catching by enemy, dis < 3 and foodcarry > 5, try to eat capsule first
        if features['oldDots'] > 3 and self.disToNearestGhost(gameState) < 3:
            newWeights['disToCapsule'] = -7.36111562824

        return features * newWeights



    ####################
    #  help functions  #
    ####################
    def isTowardsCapsule(self, previous, action):
        gameState = self.getSuccessor(previous,action)
        oldCapsule = self.getCapsules(previous)
        capsule = self.getCapsules(gameState)
        if len(oldCapsule) > len(capsule): return True
        else:
            return self.disToNearestCapsule(previous) - self.disToNearestCapsule(gameState) > 0


    def disToNearestCapsule(self, gameState):
        capsule = self.getCapsules(gameState)
        x, y = gameState.getAgentState(self.index).getPosition()
        if len(capsule) == 0:
            return 0
        else:
            dis = []
            for c in capsule:
                dis.append(self.getMazeDistance((x, y), c))
            return min(dis)


    def disToNearestGhost(self, gameState):
        agentPosition = gameState.getAgentState(self.index).getPosition()
        enemies = []
        for e in self.getOpponents(gameState):
            enemyState = gameState.getAgentState(e)
            if not enemyState.isPacman and not enemyState.getPosition() is None and not enemyState.scaredTimer > 5:
                enemies.append(enemyState)
        if len(enemies) > 0:
            toEnemies = []
            for e in enemies:
                enemyPos = e.getPosition()
                toEnemies.append(self.getMazeDistance(agentPosition, enemyPos))
            dis = min(toEnemies)
            if dis > 6:
                dis = 6
            return dis
        else:
            return 6

    def atCenter(self, myPos):
        if myPos in self.boundary:
            return True
        else:
            return False

    # return a list of (x,y) shows the food lost in our side
    def foodLostPosition(self):
        currentState = self.getCurrentObservation()
        previousState = self.getPreviousObservation()
        currentFood = self.getFoodYouAreDefending(currentState)
        self.currFoodNum = len(currentFood.asList())
        previousFood = None
        if previousState != None:
            previousFood = self.getFoodYouAreDefending(previousState)
            self.preFoodNum = len(previousFood.asList())
            if not currentFood == previousFood and self.currFoodNum < self.preFoodNum:
                food = self.differInMatrax(currentFood, previousFood)
                return food

    # gice the difference element of two matrax
    def differInMatrax(self, current, previous):
        list_curr = []
        list_pre = []
        foodPos = []
        for i in current:
            list_curr.append(i)
        for i in previous:
            list_pre.append(i)

        for i in range(len(list_pre)):
            if not list_pre[i] == list_curr[i]:
                for j in range(len(list_curr[i])):
                    if not list_curr[i][j] == list_pre[i][j]:
                        foodPos.append((i, j))

        return foodPos

    # where enemy will arrive
    def enemyArrayPoint(self, gameState):
        currentState = self.getCurrentObservation()
        food = self.getFoodYouAreDefending(currentState).asList()
        if self.index - 1 >= 0:
            self.startBlue = gameState.getInitialAgentPosition(self.index - 1)
        else:
            self.startBlue = gameState.getInitialAgentPosition(self.index + 1)
        dis = None
        if len(food) > 2:
            minDis = 1000
            minGate = []
            indexGate = []
            test = []
            for a in self.boundaryBlue:
                minGate.append(self.getMazeDistance(a,self.startBlue))
                test.append((a,self.getMazeDistance(a,self.startBlue)))
            minGate.sort()
            if len(minGate)>2:
                minGate = minGate[:2]
            else:
                minGate = minGate[0]

            for a in self.boundaryBlue:
                for i in minGate:
                    if i == self.getMazeDistance(a, self.startBlue):
                        indexGate.append(a)
            for f in food:
                for a in indexGate:
                    currDis = self.getMazeDistance(a,f)
                    if currDis <= minDis:
                        minDis = currDis
                        dis = a
        return dis


    ##################
    # min-max search #
    ##################
    def evaluate_minimax(self, gameState, agentType):
        actions = gameState.getLegalActions(self.index)
        values = [self.evaluate(gameState, a, agentType) for a in actions]
        maxValue = max(values)
        return maxValue

    def minMaxValue(self, gameState, agentIndex, alpha, beta, depth, agentType):
        if depth == 0 or gameState.isOver():
            return self.evaluate_minimax(gameState, agentType)

        if agentIndex == self.index:
            return self.maxValue(gameState, agentIndex, alpha, beta, depth, agentType)

        elif gameState.isOnRedTeam(agentIndex) != gameState.isOnRedTeam(self.index) \
                and gameState.getAgentPosition(agentIndex) != None \
                and not gameState.getAgentState(agentIndex).isPacman:
            return self.minValue(gameState, agentIndex, alpha, beta, depth, agentType)
        else:
            return self.minMaxValue(gameState, (agentIndex + 1) % 4, alpha, beta, depth, agentType)

    def maxValue(self, gameState, agentIndex, alpha, beta, depth, agentType):
        v = -float('inf')
        for a in gameState.getLegalActions(agentIndex):
            s = gameState.generateSuccessor(agentIndex, a)
            v = max(v, self.minMaxValue(s, ((agentIndex + 1) % 4), alpha, beta, depth - 1, agentType))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def minValue(self, gameState, agentIndex, alpha, beta, depth, agentType):
        v = float('inf')
        past_dist = self.getMazeDistance(gameState.getAgentPosition(self.index), gameState.getAgentPosition(agentIndex))
        for a in gameState.getLegalActions(agentIndex):
            s = gameState.generateSuccessor(agentIndex, a)
            if self.getMazeDistance(s.getAgentPosition(self.index), gameState.getAgentPosition(self.index)) >= 2 and s.getAgentPosition(self.index) == s.getInitialAgentPosition(self.index):
                return -float('inf')
            if self.getMazeDistance(s.getAgentPosition(self.index),
                                    s.getAgentPosition(agentIndex)) < past_dist or self.getMazeDistance(s.getAgentPosition(self.index),
                                                                                                        s.getAgentPosition(agentIndex)) <= 2:
                v = min(v, self.minMaxValue(s, ((agentIndex + 1) % 4), alpha, beta, depth - 1, agentType))
                if v <= alpha:
                    return v
                beta = min(beta, v)
        return v





class OffensiveReflexAgent(ReflexCaptureAgent):

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        epislon = 0  # the chanse to randomly choose an action - going to 0 at last

        '''
        #############
        # MCTsearch #
        #############
        We decided not to use in our final submission, since it is not works well
        '''
        # return MCTsearch(gameState, self, depth=5)

        # agent type always be offence
        agentType = 'offence'
        actions = gameState.getLegalActions(self.index)
        actions.remove(Directions.STOP)

        if util.flipCoin(epislon):
            action = random.choice(actions)
            # self.updateWeights(gameState, action)
            return action

        # add stop action
        if self.disToNearestGhost(gameState) == 2:
            survive_moves = len(actions)
            for action in actions:
                features = self.getFeaturesOffense(gameState, action)
                if features['disToGhost'] == 1 or features['deadends']==100:
                    survive_moves -= 1
            if survive_moves == 0:
                return Directions.STOP


        maxQ = -float("inf")
        maxQaction = None
        for action in actions:
            qval = self.evaluate(gameState, action, agentType)
            if qval >= maxQ:
                maxQ = qval
                maxQaction = action

        '''
        ##############
        # Q-learning #
        ##############
        We use approximate Q-learning and update the weights
        '''
        # self.updateWeights(gameState, maxQaction)
        return maxQaction


    ##############
    # Q learning #
    ##############
    def getReward(self, gameState, action):
        reward = 0
        nextState = self.getSuccessor(gameState, action)
        score = nextState.getScore() - gameState.getScore()
        stepCost = -0.2
        foodReward = 1
        disToGhost = self.disToNearestGhost(gameState)
        food = self.getFood(gameState)
        dx, dy = nextState.getAgentState(self.index).getPosition()
        dots = gameState.getAgentState(self.index).numCarrying

        if food[int(dx)][int(dy)]:
            reward += foodReward

        capsule = self.getCapsules(nextState)
        if len(capsule) and capsule[0] == (dx, dy):
            reward += 10
        if disToGhost <= 2:
            reward += -20 + dots * -2

        agentPosition = gameState.getAgentState(self.index).getPosition()
        if self.deadEnds.has_key((agentPosition, action)) and self.deadEnds[(agentPosition, action)] * 2 > disToGhost:
            reward -= 200

        disToBoundary = 99999
        for a in range(len(self.boundary)):
            disToBoundary = min(disToBoundary, self.getMazeDistance(agentPosition, self.boundary[a]))

        return reward + stepCost + 20 * score - 1 * disToBoundary * dots


    def getMaxQ(self, gameState):
        Qvalue = []
        actions = gameState.getLegalActions(self.index)
        for a in actions:
            Qvalue.append(self.evl2(gameState, a))
        return max(Qvalue)

    def updateWeights(self, gameState, action):
        alpha = 0.0001
        discount = 0.7
        nextState = self.getSuccessor(gameState, action)
        features = self.getFeaturesOffense(gameState, action)

        reward = self.getReward(gameState, action)
        maxQ = self.getMaxQ(nextState)
        q = self.evl2(gameState, Directions.STOP)
        for f in features:
            self.weights[f] += alpha * (reward + discount * maxQ - q) * features[f]


class DefensiveReflexAgent(ReflexCaptureAgent):
    # food lost to store lost food position
    # until wait time become 0, the agent won't offence
    foodLost = []
    waitTime = 60
    arrive = False

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        epislon = 0  # the chanse to randomly choose an action - going to 0 at last
        self.disScaredGhost = 0

        actions = gameState.getLegalActions(self.index)
        actions.remove(Directions.STOP)
        myPos = gameState.getAgentPosition(self.index)
        opponents = self.getOpponents(gameState)
        enemyDis = []
        for i in opponents:
            if not gameState.getAgentPosition(i) == None:
                enemyDis.append(self.getMazeDistance(gameState.getAgentPosition(i), myPos))
                if gameState.getAgentState(i).isPacman and gameState.getAgentState(self.index).scaredTimer >0:
                    self.disScaredGhost = self.getMazeDistance(myPos, gameState.getAgentPosition(i))


        """
        switch agent type here
        """
        agentType = 'offence'

        # Predict the location where the enemy will arrive
        if myPos in self.boundary:
            self.arrive = True
        if self.arrive == False and self.waitTime > 2:
            agentType = 'move'

        # the begining 40 steps won't attack
        self.waitTime -= 1
        if myPos in self.boundary and self.waitTime > 2:
            agentType = 'defence'

        # if we see enemy, we won't offence
        if not enemyDis == []:
            for i in enemyDis:
                if 0<= i <= 3 :
                    agentType = 'defence'

        # if there is enemy in our filed, we defence
        for enemy in opponents:
            if gameState.getAgentState(enemy).isPacman:
                agentType = 'defence'

        '''
        ##################
        # Min-Max needed #
        ##################
        We decided not to use in our final submission, since it is not works well
        '''
        # if self.disToNearestGhost(gameState) <=5 and agentType == "offence":
        #     bestScore, alpha = -float('inf'), -float('inf')
        #     action = None
        #     for a in actions:
        #         score = self.minMaxValue(gameState.generateSuccessor(self.index, a), self.index,
        #                                  alpha, bestScore, 4, agentType)
        #         if score >= bestScore:
        #             bestScore = score
        #             action = a
        #         if score > alpha:
        #             alpha = score
        #     return action

        # when go back home, aviod be eaten by ghost
        if self.disScaredGhost == 2 :
            survive_moves = len(actions)
            for action in actions:
                features = self.getFeaturesDefence(gameState, action)
                if features['dangerDistance'] == 1:
                    survive_moves -= 1
            if survive_moves == 0:
                return Directions.STOP

        if util.flipCoin(epislon):
            action = random.choice(actions)
            return action

        maxQ = -float("inf")
        maxQaction = None
        for action in actions:
            qval = self.evaluate(gameState, action, agentType)
            if qval >= maxQ:
                maxQ = qval
                maxQaction = action
        return maxQaction



# Takes a coord and a direction(NEWS), returns the next position and the reverse direction
def nextStep((x, y), direction):
    if direction == Directions.SOUTH:
        return ((x, y - 1), Directions.NORTH)
    elif direction == Directions.NORTH:
        return ((x, y + 1), Directions.SOUTH)
    elif direction == Directions.EAST:
        return ((x + 1, y), Directions.WEST)
    else:
        return ((x - 1, y), Directions.EAST)


def getDeadEnds(gameState, isRed):
    # need to consider food and depth

    deadEnds = {}
    neighbors = {}
    walkable = []

    for i in range(1, gameState.data.layout.width - 1):
        for j in range(1, gameState.data.layout.height - 1):
            if not gameState.hasWall(i, j):
                walkable.append((i, j))

    for (i, j) in walkable:
        neighbor = []
        if (i + 1, j) in walkable:
            neighbor.append(Directions.EAST)
        if (i - 1, j) in walkable:
            neighbor.append(Directions.WEST)
        if (i, j + 1) in walkable:
            neighbor.append(Directions.NORTH)
        if (i, j - 1) in walkable:
            neighbor.append(Directions.SOUTH)

        neighbors[(i, j)] = neighbor

    for (i, j) in neighbors:
        if len(neighbors[(i, j)]) >= 3:
            for direction in neighbors[(i, j)]:
                (i1, j1), revdir = nextStep((i, j), direction)
                nextNeighbor = neighbors[(i1, j1)]
                if len(nextNeighbor) >= 3:
                    continue
                elif len(nextNeighbor) == 1:
                    deadEnds[(i, j), direction] = 1


                else:
                    depth = 1
                    while len(nextNeighbor) == 2:
                        depth += 1
                        nextNeighbor.remove(revdir)
                        nextNeighbor.append(revdir)
                        (i1, j1), revdir = nextStep((i1, j1), nextNeighbor[0])
                        nextNeighbor = neighbors[(i1, j1)]
                        if len(nextNeighbor) >= 3:
                            continue
                        elif len(nextNeighbor) == 1:
                            deadEnds[(i, j), direction] = depth

    hasNew = True
    while hasNew:
        hasNew = False

        deadEnd_coords = {}
        deadEnd_potential = []

        for (i, j), dir in deadEnds:
            if not deadEnd_coords.has_key((i, j)):
                deadEnd_coords[(i, j)] = 0
            else:
                deadEnd_potential.append((i, j))  # potential: all coords that points 2 directions at a deadend
            deadEnd_coords[(i, j)] += deadEnds[(i, j), dir]

        for (i, j) in deadEnd_potential:
            waystodeadend = []
            for neighbor in neighbors[(i, j)]:
                if ((i, j), neighbor) not in deadEnds:  # i,j pointing at non-deadend directions
                    waystodeadend.append(nextStep((i, j), neighbor))  # append the nextstep and reverse dir

            if len(waystodeadend) == 1:
                (x, y), direction = waystodeadend[0]
                if ((x, y), direction) not in deadEnds:
                    hasNew = True
                    newDepth = 1 + deadEnd_coords[(i, j)]
                    deadEnds[(x, y), direction] = newDepth


                    hasAnotherNew = True
                    while hasAnotherNew:
                        hasAnotherNew = False
                        waysToAnotherDeadend = []
                        for neighbor in neighbors[
                            (x, y)]:  # a new deadend, if only one direction goes to it, then found another deadend
                            if ((x, y), neighbor) not in deadEnds:
                                waysToAnotherDeadend.append(nextStep((x, y), neighbor))
                        if len(waysToAnotherDeadend) == 1:
                            (x, y), direction = waysToAnotherDeadend[0]
                            if ((x, y), direction) not in deadEnds:
                                hasAnotherNew = True
                                newDepth += 1
                                deadEnds[(x, y), direction] = newDepth

    if isRed:
        cX = (gameState.data.layout.width - 2) / 2
        deadEnds = dict(
            (((x, y), dir), deadEnds[((x, y), dir)]) for ((x, y), dir) in deadEnds if x > cX)  # deadend filter
    else:
        cX = ((gameState.data.layout.width - 2) / 2) + 1
        deadEnds = dict(
            (((x, y), dir), deadEnds[((x, y), dir)]) for ((x, y), dir) in deadEnds if x < cX)  # deadend filter

    return deadEnds


##########
# MCTree #
##########

class MCTree:
    def __init__(self, gameState, ancestor=None, agent=None):
        self.gameState = gameState
        self.subtrees = {}
        # self.subtreesCounter = {}
        self.ancestor = ancestor
        self.depth = 0
        self.visited = 0
        self.score = 0
        self.agent = agent
        self.agentindex = agent.index

        if ancestor is not None:
            self.depth = ancestor.depth + 1

    def __str__(self):
        return ""

    def expand(self):
        if len(self.subtrees) == 0:
            for action in self.gameState.getLegalActions(self.agentindex):
                self.subtrees[action] = MCTree(self.gameState.generateSuccessor(self.agentindex, action), self,
                                               self.agent)

    def tree_policy(self):
        actions = self.gameState.getLegalActions(self.agentindex)
        if len(actions) == 1:
            return actions[0]

        maxQ = -10000
        maxQaction = Directions.STOP
        CP = 1 / 2
        for action in actions:
            # if self.subtreesCounter[action] == 0:
            #     self.subtreesCounter[action] += 1
            if self.subtrees[action].visited == 0:
                return action
            uct = self.agent.evl(self.gameState.generateSuccessor(self.agentindex, action)) + 2 * CP * sqrt(
                2 * log(self.visited) / self.subtrees[action].visited)
            if uct >= maxQ:
                maxQ = uct
                maxQaction = action
        return maxQaction

    def backprop(self, simulate_score):
        self.visited += 1
        self.score += simulate_score
        if self.ancestor is not None:
            self.ancestor.backprop(simulate_score)


def random_simulation(gameState, agent, depth):
    agentindex = agent.index
    if depth > 0:
        actions = gameState.getLegalActions(agentindex)
        reverse_direction = Directions.REVERSE[gameState.getAgentState(agentindex).getDirection()]
        if len(actions) > 1:
            actions.remove(reverse_direction)
            action = random.choice(actions)
        else:
            action = actions[0]
        random_simulation(gameState.generateSuccessor(agentindex, action), agent, depth - 1)
    return agent.evl(gameState)


def MCTsearch(gameState, agent, depth):
    start_time = time.time()
    root = MCTree(gameState, None, agent)
    root.expand()

    while time.time() - start_time < 0.08:  # budget time
        action = root.tree_policy()
        tree = root.subtrees[action]
        while not tree.visited == 0:
            tree.expand()  # effective if len(subtrees) == 0
            action = tree.tree_policy()
            tree = tree.subtrees[action]
        simulated_score = random_simulation(tree.gameState, agent, depth)
        tree.backprop(simulated_score)

    maxQ = -10000
    maxQaction = None
    for action in root.gameState.getLegalActions(root.agentindex):
        subtree = root.subtrees[action]
        if subtree.visited == 0:
            continue
        qvalue = subtree.score / subtree.visited
        if qvalue >= maxQ:
            maxQ = qvalue
            maxQaction = action
    # exit(100)
    return maxQaction


