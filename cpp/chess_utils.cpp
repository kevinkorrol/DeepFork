#include <iostream>
#include <pybind11/>
#include <torch/script.h>
#include <chess.hpp>

namespace py = pybind11;
using namespace chess;
using namespace torch;

constexpr std::array<PieceType, 6> PIECE_TYPES = {
    PieceType::PAWN,
    PieceType::KNIGHT,
    PieceType::BISHOP,
    PieceType::ROOK,
    PieceType::QUEEN,
    PieceType::KING
};

void test_library() {
    const auto board = Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    Movelist moves;
    movegen::legalmoves(moves, board);

    for (const auto& move : moves) {
        std::cout << uci::moveToUci(move) << "\n";
    }
}

void add_pieces_to_tensor(
    const PieceType piece,
    const Color player,
    const bool is_black_turn,
    float* plane,
    const Board& board,
    const int plane_index
    ) {
    Bitboard piece_bits = board.pieces(piece, player);
    while (piece_bits) {
        Square bit = piece_bits.pop();
        const int square = bit.index();
        const int square_int = is_black_turn ? (square ^ 56) : square;
        plane[plane_index * 64 + square_int] = 1.0f;
    }
}

Tensor get_piece_placement_planes(const Board& board, const bool is_black_turn) {
    const auto options = TensorOptions().dtype(kFloat32);
    Tensor planes = torch::zeros({12, 8, 8}, options);

    float* current_plane = planes.data_ptr<float>();

    const auto player = is_black_turn ? Color::BLACK : Color::WHITE;
    const auto opponent = is_black_turn ? Color::WHITE : Color::BLACK;

    int plane_index = 0;

    for (const auto piece : PIECE_TYPES) {
        add_pieces_to_tensor(piece, player, is_black_turn, current_plane, board, plane_index);
        add_pieces_to_tensor(piece, opponent, is_black_turn, current_plane, board, plane_index + 6);
        plane_index++;
    }
    
    return planes;
}

Tensor get_repetition_counter_planes(const Board& board) {
    const auto options = TensorOptions().dtype(kFloat32);
    Tensor planes = torch::zeros({2, 8, 8}, options);

    if (board.isRepetition(1)) {
        planes[0].fill_(1.0f);
    }

    if (board.isRepetition(2)) {
        planes[1].fill_(1.0f);
    }

    return planes;
}

Tensor get_global_planes(const Board& board, const bool is_black_turn) {
    const auto options = TensorOptions().dtype(kFloat32);
    Tensor global_planes = torch::zeros({7, 8, 8}, options);

    constexpr auto king_side = Board::CastlingRights::Side::KING_SIDE;
    constexpr auto queen_side = Board::CastlingRights::Side::QUEEN_SIDE;
    const auto player_color = is_black_turn ? Color::BLACK : Color::WHITE;
    const auto opponent_color = is_black_turn ? Color::WHITE : Color::BLACK;

    if (!is_black_turn) {
        global_planes[0].fill_(1.0);
    }

    if (board.castlingRights().has(player_color, king_side)) {
        global_planes[1].fill_(1.0f);
    }
    if (board.castlingRights().has(player_color, queen_side)) {
        global_planes[2].fill_(1.0f);
    }
    if (board.castlingRights().has(opponent_color, king_side)) {
        global_planes[3].fill_(1.0f);
    }
    if (board.castlingRights().has(opponent_color, queen_side)) {
        global_planes[4].fill_(1.0f);
    }

    const auto move_count = static_cast<float>(board.fullMoveNumber());
    global_planes[5].fill_(move_count / 200.0f);

    const auto half_clock = static_cast<float>(board.halfMoveClock());
    global_planes[6].fill_(half_clock / 50.0f);

    return global_planes;
}

Tensor get_legal_moves_plane(const Board& board, bool is_black_turn) {
    const auto options = TensorOptions().dtype(kFloat32);
    Tensor plane = torch::zeros({1, 8, 8}, options);

    Movelist moves;
    movegen::legalmoves(moves, board);

    float* data = plane.data_ptr<float>();

    for (const auto& move : moves) {
        Square square = move.to();
        int rank = square.rank();

        if (is_black_turn) {
            rank = 7 - rank;
        }
        data[rank * 8 + square.file()] = 1.0f;
    }

    return plane;
}

void update_state_history(
    const Board& board,
    Tensor &state_history,
    const int history_count,
    const bool is_black_turn
    ) {
    for (int i = history_count - 1; i > 0; i--) {
        state_history[i] = state_history[i - 1];
    }
    state_history[0] = cat({
        get_repetition_counter_planes(board),
        get_piece_placement_planes(board, is_black_turn)
        });
}

Tensor state_to_tensor(Tensor state_history, const Board& board, int history_count) {
    const bool is_black_turn = board.sideToMove() == Color::BLACK;
    Tensor global_planes = get_global_planes(board, is_black_turn);
    Tensor legal_moves_plane = get_legal_moves_plane(board, is_black_turn);
    update_state_history(board, state_history, history_count, is_black_turn);

    Tensor flat_history = state_history.reshape({-1, 8, 8});

    return cat({flat_history, global_planes, legal_moves_plane}, 0);
}

// ... (Your existing functions go here) ...

void print_tensor_stats(const Tensor& t, std::string name) {
    std::cout << "--- " << name << " ---\n";
    std::cout << "Shape: " << t.sizes() << "\n";
    std::cout << "Sum:   " << t.sum().item<float>() << "\n"; // Quick check if planes are empty
    std::cout << "Max:   " << t.max().item<float>() << "\n\n";
}

int main() {
    std::cout << "=== STARTING CHESS TENSOR TEST ===\n\n";

    // 1. Initialize Board
    Board board = Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    std::cout << "Initial FEN: " << board.getFen() << "\n\n";

    // 2. Initialize History Tensor
    // History Shape: [T=8, Features=14, H=8, W=8]
    // 14 Features = 2 Repetition + 12 Piece Planes
    const int history_steps = 8;
    Tensor state_history = torch::zeros({history_steps, 14, 8, 8}, kFloat32);

    // ============================================================
    // TURN 1: WHITE TO MOVE
    // ============================================================
    std::cout << ">>> Generating Tensor for Turn 1 (White)...\n";
    Tensor input_tensor = state_to_tensor(state_history, board, history_steps);

    // EXPECTED SHAPE:
    // History (8*14 = 112) + Global (7) + Legal (1) = [120, 8, 8]
    print_tensor_stats(input_tensor, "Input Tensor T1");

    float e2_val = input_tensor[2][1][4].item<float>();
    std::cout << "Check: White Pawn on E2 (Rank 1, File 4): " << e2_val << " (Expect 1.0)\n\n";


    // ============================================================
    // MAKE MOVE: e2e4
    // ============================================================
    Move move1 = uci::parseSan(board, "e4");
    std::cout << ">>> Playing Move: " << uci::moveToUci(move1) << "\n";
    board.makeMove(move1);


    // ============================================================
    // TURN 2: BLACK TO MOVE (Testing Flip Logic)
    // ============================================================
    std::cout << ">>> Generating Tensor for Turn 2 (Black)...\n";
    input_tensor = state_to_tensor(state_history, board, history_steps);

    print_tensor_stats(input_tensor, "Input Tensor T2");

    float e4_flipped_val = input_tensor[8][4][4].item<float>();
    std::cout << "Check: Opponent Pawn (White) on e4.\n";
    std::cout << "       From Black's perspective, this is Rank 4, File 4.\n";
    std::cout << "       Value: " << e4_flipped_val << " (Expect 1.0)\n\n";


    // ============================================================
    // MAKE MOVE: e7e5 (Black responds)
    // ============================================================
    Move move2 = uci::parseSan(board, "e5");
    std::cout << ">>> Playing Move: " << uci::moveToUci(move2) << "\n";
    board.makeMove(move2);


    // ============================================================
    // TURN 3: WHITE TO MOVE (History Check)
    // ============================================================
    std::cout << ">>> Generating Tensor for Turn 3 (White)...\n";
    input_tensor = state_to_tensor(state_history, board, history_steps);


    float hist_e2_val = input_tensor[28+2][1][4].item<float>();
    std::cout << "Check: History T-2 (Start Position).\n";
    std::cout << "       Looking for White Pawn on E2 in history planes.\n";
    std::cout << "       Value: " << hist_e2_val << " (Expect 1.0)\n";

    std::cout << "\n=== TEST COMPLETE ===\n";
    return 0;
}


PYBIND11_MODULE(deepfork_cpp, m) {
    m.doc() = "DeepFork Chess Tensor Engine";

    m.def("state_to_tensor",
        [](Tensor state_history, std::string fen, int history_count) {
            Board board(fen);

            return state_to_tensor(state_history, board, history_count);
        },
        "Generates the AlphaZero input tensor from a FEN string and history buffer.",
        py::arg("state_history"),
        py::arg("fen"),
        py::arg("history_count")
    );
}