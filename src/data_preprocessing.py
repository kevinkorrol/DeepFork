import chess.pgn

pgn = open('/home/tonis/Documents/25sügis/sjandmeteadusesse/DeepFork/data/raw/lichess_db_standard_rated_2013-01.pgn')

game = chess.pgn.read_game(pgn)

board = game.board()
for move in game.mainline_moves():
    board.push(move)

board