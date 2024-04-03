# multiAgents.py
# --------------
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


import math
import random

from game import Directions
from game import Agent
from pacman import GameState
from util import manhattanDistance
import util
class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = bestIndices[0] # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]
    
    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)

        penalty = 0
        if action == 'Stop':
            penalty = -100

        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        ghostCloseness = [manhattanDistance(newPos, g.getPosition()) for g in newGhostStates]
        closestGhost = min(ghostCloseness)
        
        newFoodCloseness = [manhattanDistance(newPos, food) for food in newFood.asList()]
        
        if len(newFoodCloseness) == 0:
            return 1000


        return successorGameState.getScore() + (closestGhost) / (10 * min(newFoodCloseness)) + penalty
    
def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """

        return self.maxormin(gameState, 0, 0)[1]
        
    def maxormin(self, gameState, agentIndex, ply): 
        bestEval = -math.inf if agentIndex == 0 else math.inf
        bestAction = None

        if gameState.isWin() or gameState.isLose() or ply == self.depth * gameState.getNumAgents():
            return self.evaluationFunction(gameState), None

        for action in gameState.getLegalActions(agentIndex):
            state = gameState.generateSuccessor(agentIndex, action)

            e, a = self.maxormin(state, (agentIndex + 1) % gameState.getNumAgents(), ply + 1)
    
            if agentIndex == 0: # maximize
                if e > bestEval:
                    bestEval = e
                    bestAction = action
            else: # minimize
                if e < bestEval:
                    bestEval = e
                    bestAction = action
            

        return bestEval, bestAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        return self.alphabeta(gameState, 0, 0, -math.inf, math.inf)[1]

    def alphabeta(self, gameState, agentIndex, ply, alpha, beta):
        bestEval = -math.inf if agentIndex == 0 else math.inf
        bestAction = None

        if gameState.isWin() or gameState.isLose() or ply == self.depth * gameState.getNumAgents():
            return self.evaluationFunction(gameState), None

        for action in gameState.getLegalActions(agentIndex):
            state = gameState.generateSuccessor(agentIndex, action)

            e, a = self.alphabeta(state, (agentIndex + 1) % gameState.getNumAgents(), ply + 1, alpha, beta)

            if agentIndex == 0: # maximize
                if e > bestEval:
                    bestEval, bestAction = e, action
                if bestEval > beta:
                    return bestEval, bestAction
                alpha = max(bestEval, alpha)
            else: # minimize
                if e < bestEval:
                    bestEval, bestAction = e, action
                if bestEval < alpha: 
                    return bestEval, bestAction
                beta = min(bestEval, beta)

        return bestEval, bestAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        return self.expectimax(gameState, 0, 0)[1]

    def expectimax(self, gameState, agentIndex, ply): 
        bestEval = -math.inf if agentIndex == 0 else math.inf
        bestAction = None

        # if terminal state return evaluation and no move
        if gameState.isWin() or gameState.isLose() or ply == self.depth * gameState.getNumAgents():
            return self.evaluationFunction(gameState), None

        # if maximizing
        if agentIndex == 0: 
            for action in gameState.getLegalActions(agentIndex):
                state = gameState.generateSuccessor(agentIndex, action)
                e, a = self.expectimax(state, (agentIndex + 1) % gameState.getNumAgents(), ply + 1)
                if e > bestEval:
                    bestEval = e
                    bestAction = action
        else: # expectimax
            bestEval = sum(
                        (self.expectimax(gameState.generateSuccessor(agentIndex, action), (agentIndex + 1) % gameState.getNumAgents(), ply + 1)[0]
                        for action in gameState.getLegalActions(agentIndex))
                    ) / len(gameState.getLegalActions(agentIndex)) 

        return bestEval, bestAction

def betterEvaluationFunction(gameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    newPos = gameState.getPacmanPosition()
    newFood = gameState.getFood()
    newGhostStates = gameState.getGhostStates()
    newFoodCloseness = [manhattanDistance(newPos, food) for food in newFood.asList()]
    ghost = gameState.getGhostStates()[0]
    ghostCloseness = [manhattanDistance(newPos, g.getPosition()) for g in newGhostStates]
    closestGhost = min(ghostCloseness)

    if len(newFoodCloseness) == 0:
        return 1000
    
    if ghost.scaredTimer != 0:
        return 1000 - ghostCloseness[0]

    return gameState.getScore() + (closestGhost) / (10 * min(newFoodCloseness))  


# Abbreviation
better = betterEvaluationFunction
