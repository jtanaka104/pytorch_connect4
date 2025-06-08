import streamlit as st
import torch
import torch.nn as nn
import numpy
import time
import pathlib

# --- 提供されたソースコードのクラス定義 ---
# 【修正点A】元の学習コードの仕様に合わせるため、get_observationを元に戻します。
# これにより、モデルは学習時と同じ形式のデータを常に受け取ります。

class Connect4:
    """ Connect4のゲーム環境を管理するクラス """
    def __init__(self):
        self.board = numpy.zeros((6, 7), dtype="int32")
        self.player = 1  # 1: 先手 (人間), -1: 後手 (AI)

    def get_player(self):
        return self.player

    def reset(self):
        self.board = numpy.zeros((6, 7), dtype="int32")
        self.player = 1

    def step(self, action):
        if self.board[5][action] != 0:
            return self.get_observation(), 0, False

        for i in range(6):
            if self.board[i][action] == 0:
                self.board[i][action] = self.player
                break
        
        current_player_won = self.have_winner()
        done = current_player_won or len(self.legal_actions()) == 0
        reward = 1 if current_player_won else 0
        
        # プレイヤー交代
        self.player *= -1

        return self.get_observation(), reward, done

    def get_observation(self):
        # 元の学習コードの仕様に合わせ、視点変換は行わない
        board_player1 = numpy.where(self.board == 1, 1, 0)
        board_player2 = numpy.where(self.board == -1, 1, 0)
        return numpy.array([board_player1, board_player2], dtype="int32").flatten()

    def legal_actions(self):
        legal = []
        for i in range(7):
            if self.board[5][i] == 0:
                legal.append(i)
        return legal

    def have_winner(self):
        player_to_check = self.player
        
        # Horizontal check
        for i in range(4):
            for j in range(6):
                if (self.board[j][i] == player_to_check and
                    self.board[j][i + 1] == player_to_check and
                    self.board[j][i + 2] == player_to_check and
                    self.board[j][i + 3] == player_to_check):
                    return True
        # Vertical check
        for i in range(7):
            for j in range(3):
                if (self.board[j][i] == player_to_check and
                    self.board[j + 1][i] == player_to_check and
                    self.board[j + 2][i] == player_to_check and
                    self.board[j + 3][i] == player_to_check):
                    return True
        # Positive diagonal check
        for i in range(4):
            for j in range(3):
                if (self.board[j][i] == player_to_check and
                    self.board[j + 1][i + 1] == player_to_check and
                    self.board[j + 2][i + 2] == player_to_check and
                    self.board[j + 3][i + 3] == player_to_check):
                    return True
        # Negative diagonal check
        for i in range(4):
            for j in range(3, 6):
                if (self.board[j][i] == player_to_check and
                    self.board[j - 1][i + 1] == player_to_check and
                    self.board[j - 2][i + 2] == player_to_check and
                    self.board[j - 3][i + 3] == player_to_check):
                    return True
        return False


class Net(nn.Module):
    """ ニューラルネットワークモデルの定義 """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(91, 182)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(182, 364)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(364, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# --- Streamlit アプリケーション用の関数 ---

@st.cache_resource
def load_model(model_path="model.cpt"):
    if not pathlib.Path(model_path).is_file():
        st.error(f"モデルファイル '{model_path}' が見つかりません。")
        st.stop()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    data = torch.load(model_path, map_location=device)
    model.load_state_dict(data['model'])
    model.eval()
    return model, device

# 【デバッグ機能】AIの思考を可視化するための関数
def get_ai_action_and_scores(env, model, device):
    """
    AIの最善手と、各手の評価値を計算します。
    """
    legal_actions = env.legal_actions()
    if not legal_actions:
        return None, {}

    states = env.get_observation()
    scores = {}
    best_action = -1
    max_predict = -float('inf')

    for action in legal_actions:
        wa = [0.] * 7
        wa[action] = 1.0
        wx = numpy.append(states, wa).astype('float32')
        tX = torch.from_numpy(wx).to(device)
        
        with torch.no_grad():
            y = model(tX)
        
        predict_value = y.item()
        scores[action] = predict_value

        if predict_value > max_predict:
            max_predict = predict_value
            best_action = action
            
    return best_action, scores

def display_board(board):
    header = "１２３４５６７"
    board_lines = []
    for row in reversed(range(6)):
        row_str = ""
        for col in range(7):
            if board[row, col] == 1:
                row_str += "●"
            elif board[row, col] == -1:
                row_str += "✕"
            else:
                row_str += "□"
        board_lines.append(row_str)
    board_body = "<br/>".join(board_lines)
    html = f'<span style="font-family:\'MS Gothic\',\'Osaka-Mono\',monospace;font-size:24px;line-height:1.1;">{header}<br/>{board_body}</span>'
    st.markdown(html, unsafe_allow_html=True)

# --- Streamlit アプリケーションのメイン処理 ---

st.title("Connect4（pytorch版）")

model, device = load_model()

if 'game' not in st.session_state:
    st.session_state.game = Connect4()
    st.session_state.game_over = False
    st.session_state.winner = None
    st.session_state.message = "あなたの番です。下のボタンから列を選んでください。"
    st.session_state.ai_scores = None # AIの評価値を保存する場所

display_board(st.session_state.game.board)
message_placeholder = st.empty()
message_placeholder.info(st.session_state.message)

if st.session_state.game_over:
    if st.session_state.winner == 1:
        st.success("🎉 あなたの勝ちです！おめでとうございます！")
    elif st.session_state.winner == -1:
        st.error("🤖 AIの勝ちです。")
    else:
        st.warning("引き分けです。")

    if st.button("新しいゲームを始める"):
        st.session_state.game = Connect4()
        st.session_state.game_over = False
        st.session_state.winner = None
        st.session_state.message = "あなたの番です。下のボタンから列を選んでください。"
        st.session_state.ai_scores = None
        st.rerun()

else:
    current_player = st.session_state.game.get_player()

    if current_player == 1: # 人間のターン
        legal_actions = st.session_state.game.legal_actions()
        # 1行目（1〜4列）
        cols1 = st.columns(4)
        for i in range(4):
            with cols1[i]:
                if st.button(f"{i+1}", key=f"col_{i}", disabled=(i not in legal_actions)):
                    # 【修正点B】step関数の返り値を正しく使うように修正
                    _, _, done = st.session_state.game.step(i)
                    st.session_state.ai_scores = None
                    if done:
                        st.session_state.game_over = True
                        st.session_state.winner = 1 if len(st.session_state.game.legal_actions()) > 0 else 0
                    st.rerun()
        # 2行目（5〜7列）
        cols2 = st.columns(3)
        for i in range(4, 7):
            with cols2[i-4]:
                if st.button(f"{i+1}", key=f"col_{i}", disabled=(i not in legal_actions)):
                    # 【修正点B】step関数の返り値を正しく使うように修正
                    _, _, done = st.session_state.game.step(i)
                    st.session_state.ai_scores = None
                    if done:
                        st.session_state.game_over = True
                        st.session_state.winner = 1 if len(st.session_state.game.legal_actions()) > 0 else 0
                    st.rerun()
    else: # AIのターン
        st.session_state.message = "AIが思考中です..."
        message_placeholder.info(st.session_state.message)

        with st.spinner("AIが思考中..."):
            time.sleep(1.0)
            ai_action, scores = get_ai_action_and_scores(st.session_state.game, model, device)
            st.session_state.ai_scores = scores # 評価値を保存

            if ai_action is not None:
                # 【修正点B】step関数の返り値を正しく使うように修正
                _, _, done = st.session_state.game.step(ai_action)
                
                if done:
                    st.session_state.game_over = True
                    # 石を置いたのがAIなので、勝者もAI
                    if len(st.session_state.game.legal_actions()) > 0:
                        st.session_state.winner = -1 # AI
                    else: # 引き分け
                        st.session_state.winner = 0

        st.session_state.message = "あなたの番です。下のボタンから列を選んでください。"
        st.rerun()