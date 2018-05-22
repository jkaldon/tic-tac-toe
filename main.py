import tensorflow as tf
import numpy as np
from keras import losses, optimizers
from keras.models import Sequential
from keras.layers import Dense
import random

np.random.seed(123)
board_state = np.array([0,0,0, 0,0,0, 0,0,0])

player1 = 1
player2 = -1
empty_spot = 0

model = Sequential()
model.add(Dense(8, activation='relu', input_dim=9))
model.add(Dense(9, activation='relu'))
model.compile(loss = losses.mean_absolute_error, optimizer=optimizers.Nadam())

class GameData:
  def __init__(self, winner, turns):
    self.winner = winner
    self.turns = turns

def train(model, games):
  inputs = []
  expected = []
  samples = random.sample(games, 10000)
  print("Sampled ", len(samples), " games of ", len(games))

  print("Processing samples...")
  i=0
  for game in samples:
    i += 1
    preceding_turn = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    for turn in game.turns:
      inputs.append(preceding_turn)
      delta = turn - preceding_turn

      #print("game ", i, " turn ", turn, " preceding_turn ", preceding_turn, " delta ", delta)

      if game.winner == 1:
        expected.append(delta)
      elif game.winner == -1:
        expected.append(-1 * delta)
      else:
        expected.append(0 * delta)

      preceding_turn = turn

  print("Training model...")
  model.fit(np.array(inputs), np.array(expected), epochs=15, batch_size=64)

def importGameData():
  with open('all-games.data') as f:
    content = f.readlines()

  print('Read ', len(content), ' lines of data from all-games.data.')
  i = 0
  games = []

  while len(content) > i:
    outcome = convertLetterToNumber(content[i][:1])
    line1 = content[i + 1]
    line2 = content[i + 2]
    line3 = content[i + 3]
    i+=4

    if not i%60000:
      print('Line ', i, ' of ', len(content), ' read...')

    turns = parseTurns(line1, line2, line3)

    games.append(GameData(outcome, turns))

  return games

def parseTurns(line1, line2, line3):
  turns_row_1 = line1.split(',')
  turns_row_2 = line2.split(',')
  turns_row_3 = line3.split(',')

  turns = []

  for i in range(len(turns_row_1)-1):
    s00 = convertLetterToNumber(turns_row_1[i][0:1])
    s01 = convertLetterToNumber(turns_row_1[i][1:2])
    s02 = convertLetterToNumber(turns_row_1[i][2:3])

    s10 = convertLetterToNumber(turns_row_2[i][0:1])
    s11 = convertLetterToNumber(turns_row_2[i][1:2])
    s12 = convertLetterToNumber(turns_row_2[i][2:3])

    s20 = convertLetterToNumber(turns_row_3[i][0:1])
    s21 = convertLetterToNumber(turns_row_3[i][1:2])
    s22 = convertLetterToNumber(turns_row_3[i][1:3])

    turns.append(np.array([s00, s01, s02, s10, s11, s12, s20, s21, s22]))

  return turns

def neuralnet_move(player1, board_state):
  results = model.predict(np.array([board_state]))
  print("results = ", results)
  return np.argmax(results)

def sign(player1, player2):
  player1 = input("What team you want to be? X or O ")
  while player1 not in ('x','X','o','O'):
    print("Invalid Choice!")
    player1 = input("What team you want to be? X or O ")
  if player1 == 'x' or player1 == 'X':
    print("Ok, X is yours!")
    player1 = 1
    player2 = -1
  else:
    print("Ok, O is yours!")
    player1 = -1
    player2 = 1
  return player1, player2


def decide_turn():
  turn = None
  while turn not in ('y','Y','n','N'):
    turn = input("Do you want to go first? ")
    if turn == 'y' or turn == 'Y':
      return 1
    elif turn == 'n' or turn == 'N':
      return 0
    else:
      print("its an invalid choice.")

def draw(a):
  #print(chr(27) + "[2J")
  print()
  print("\t ", convertNumberToLetter(a[0]), "|", convertNumberToLetter(a[1]), "|", convertNumberToLetter(a[2]))
  print("\t", "-----------")
  print("\t ", convertNumberToLetter(a[3]), "|", convertNumberToLetter(a[4]), "|", convertNumberToLetter(a[5]))
  print("\t", "-----------")
  print("\t ", convertNumberToLetter(a[6]), "|", convertNumberToLetter(a[7]), "|", convertNumberToLetter(a[8]), "\n")

def convertLetterToNumber(l):
  if l in ('x','X'):
    return 1
  elif l in ("o", "O"):
    return -1
  else:
    return 0

def convertNumberToLetter(n):
  if n == 1:
    return "X"
  elif n == -1:
    return "O"
  else:
    return " "

def congo_player1():
  print("Player 1 wins!")

def congo_player2():
  print("Player 2 wins!")

def player1_first(player1, player2, board_state):
  while check_for_winner(player1, player2, board_state) is None:
    #move = player1_move(player1, board_state)
    move = neuralnet_move(player1, board_state)
    print("player 1 takes ", move)
    board_state[int(move)] = player1
    draw(board_state)
    if check_for_winner(player1, player2, board_state) != None:
      print("winner found")
      break
    else:
      print("no winner")
      pass
    print("ummmm....i'll take..")
    p_move = machine_move(player1, player2, board_state)
    print(p_move)
    board_state[int(p_move)] = player2
    draw(board_state)
  q = check_for_winner(player1, player2, board_state)
  if q == 1:
    congo_player1()
  elif q == -1:
    congo_player2()
  else:
    print("Its tie man...")

def player2_first(player1, player2, new):
  while not check_for_winner(player1, player2, new):
    print("i'll take...")
    p_move = machine_move(player1, player2, new)
    print(p_move)
    new[p_move] = player2
    draw(new)
    if check_for_winner(player1, player2, new) != None:
        break
    else:
        pass
    move = player1_move(player1, new)
    new[int(move)] = player1
    draw(new)
  q = check_for_winner(player1, player2, new)
  if q == 1:
      congo_player1()
  elif q == -1:
      congo_player2()
  else:
      print("Cat's game...")


def check_for_winner(player1, player2, board_state):
  winning_states = ((0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6))
  for ws in winning_states:
    #print("board_state=", board_state, " board_state[ws[0]]=", board_state[ws[0]], " board_state[ws[1]]=", board_state[ws[1]], " board_state[ws[2]]=", board_state[ws[2]])
    if board_state[ws[0]] == board_state[ws[1]] == board_state[ws[2]] != empty_spot:
      winner = board_state[ws[0]]
      if winner == player1:
        return player1
      elif winner == player2:
        return player2
      if empty_spot not in board_state:
        return empty_spot
  if empty_spot not in board_state:
    return empty_spot
  return None

def player1_move(player1, board_state):
  a = input("where you want to move? ")
  while True:
    if a not in ('0','1','2','3','4','5','6','7','8'):
      print("Sorry, invalid move")
      a = input("where you want to move? ")
    elif board_state[int(a)] != empty_spot:
      print("Sorry, the place is already taken")
      a = input("where you want to move? ")
    else:
      return int(a)


def machine_move(player1, player2, board_state):
  #best = [4, 0, 2, 6, 8]
  blank = []
  for i in range(0,9):
    if board_state[i] == empty_spot:
        blank.append(i)

  for i in blank:
    board_state[i] = player2
    if check_for_winner(player1, player2, board_state) is 0:
      return i
    board_state[i] = empty_spot

  for i in blank:
    board_state[i] = player1
    if check_for_winner(player1, player2, board_state) is 1:
      return i
    board_state[i] = empty_spot

  return int(blank[random.randrange(len(blank))])

def display_instruction():
  print(chr(27) + "[2J")
  """ Displays Game Instuructions. """
  print(
  """
  Welcome to the Game...
  You will make your move known by entering a number, 0 - 8.
  The will correspond to the board position as illustrated:


                      0 | 1 | 2
                      -----------
                      3 | 4 | 5
                      -----------
                      6 | 7 | 8


  Prepare yourself, the ultimate battle is about to begin.....
  """)


def main(player1, player2, board_state):
  # to re-enable user input
  # player1, player2 = sign(player1, player2)
  #b = decide_turn()
  b = 1
  if b == 1:
    # print("Ok, you are first!")
    # print("Lets get started, here's a new board!")
    draw(board_state)
    player1_first(player1, player2, board_state)
  elif b == 0:
    print("Ok, I'll be the first!")
    print("So, lets start..")
    draw(board_state)
    player2_first(player1, player2, board_state)
  else:
    pass

games = importGameData()

train(model, games)

main(player1, player2, board_state)

