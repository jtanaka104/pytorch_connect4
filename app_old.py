import streamlit as st
import torch
import torch.nn as nn
import numpy
import time
import pathlib

# --- 提供されたソースコードのクラス定義 ---
# Streamlitアプリに必要なクラスを転記します。

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
        return self.get_observation()

    def step(self, action):
        # actionが合法手かチェック
        if self.board[5][action] != 0:
            return self.get_observation(), 0, False, True # 不正な手

        for i in range(6):
            if self.board[i][action] == 0:
                self.board[i][action] = self.player
                break
        
        # 勝利判定の前に、現在のプレイヤーを保存
        current_player_won = self.have_winner()
        
        done = current_player_won or len(self.legal_actions()) == 0
        reward = 1 if current_player_won else 0
        
        # プレイヤー交代
        self.player *= -1

        return self.get_observation(), reward, done, False

    def get_observation(self):
        # 現在のプレイヤーの視点での盤面を返す
        # AI(player=-1)のターンの場合、AIの石が1、人間の石が-1になるように盤面を反転させる
        if self.player == 1:
            # 人間のターン：人間が1、AIが-1
            board_player1 = numpy.where(self.board == 1, 1, 0)
            board_player2 = numpy.where(self.board == -1, 1, 0)
        else: # self.player == -1
            # AIのターン：AIが1、人間が-1
            board_player1 = numpy.where(self.board == -1, 1, 0)
            board_player2 = numpy.where(self.board == 1, 1, 0)
        return numpy.array([board_player1, board_player2], dtype="int32").flatten()

    def legal_actions(self):
        legal = []
        for i in range(7):
            if self.board[5][i] == 0:
                legal.append(i)
        return legal

    def have_winner(self):
        # 判定対象は石を置いた直後のプレイヤー
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
    """
    学習済みモデルを読み込み、キャッシュします。
    モデルファイルが見つからない場合はエラーを表示します。
    """
    if not pathlib.Path(model_path).is_file():
        st.error(f"モデルファイル '{model_path}' が見つかりません。")
        st.stop()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    
    # CPU上でモデルを読み込むために map_location を指定
    data = torch.load(model_path, map_location=device)
    model.load_state_dict(data['model'])
    model.eval()  # 推論モードに設定
    
    return model, device

def get_ai_action(env, model, device):
    """
    AIの最善手を計算します。
    モデルの入力仕様に基づき、全ての合法手について評価値を取得し、
    最も評価値の高い手を選択します。
    """
    legal_actions = env.legal_actions()
    if not legal_actions:
        return None

    # AIの視点での盤面状態を取得
    states = env.get_observation()
    
    best_action = -1
    max_predict = -999999

    # 各合法手に対して評価値を計算
    for action in legal_actions:
        # モデルの入力を作成 (盤面状態 + action)
        wa = [0.] * 7
        wa[action] = 1.0
        wx = numpy.append(states, wa).astype('float32')
        
        # PyTorchテンソルに変換
        tX = torch.from_numpy(wx).to(device)
        
        # モデルで推論
        with torch.no_grad():
            y = model(tX)
        
        predict_value = y.item()
        print("action:", action, "predict_value:", int(predict_value*100))

        if predict_value > max_predict:
            max_predict = predict_value
            best_action = action
    print()
    return best_action

def display_board(board):
    """
    現在の盤面をStreamlitに表示します。
    - ●: あなた (先手)
    - Ｘ: AI (後手)
    - □: 空
    """
    header = "１２３４５６７"
    board_lines = []
    for row in reversed(range(6)):
        row_str = ""
        for col in range(7):
            if board[row, col] == 1:
                row_str += "●"
            elif board[row, col] == -1:
                row_str += "Ｘ"
            else:
                row_str += "□"
        board_lines.append(row_str)
    # <br/>で改行
    board_html = header + "<br/>" + "<br/>".join(board_lines)
    st.markdown(
        f'<span style="font-family: &quot;MS Gothic&quot;, &quot;Osaka-Mono&quot;, &quot;monospace&quot;; font-size: 20px;">{board_html}</span>',
        unsafe_allow_html=True
    )


# --- Streamlit アプリケーションのメイン処理 ---

st.title("Connect4（Pytorch版）")

# モデルの読み込み
model, device = load_model()

# --- セッション状態の初期化 ---
if 'game' not in st.session_state:
    st.session_state.game = Connect4()
    st.session_state.game_over = False
    st.session_state.winner = None
    st.session_state.message = "あなたの番です。下のボタンから列を選んでください。"

# --- 盤面とメッセージの表示 ---
display_board(st.session_state.game.board)
message_placeholder = st.empty()
message_placeholder.info(st.session_state.message)


# --- ゲーム進行のメインロジック ---
if st.session_state.game_over:
    # --- ゲーム終了時の処理 ---
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
        st.rerun()

else:
    # --- ゲーム進行中の処理 ---
    is_human_turn = st.session_state.game.get_player() == 1
    
    if is_human_turn:
        # --- 人間のターン ---
        legal_actions = st.session_state.game.legal_actions()
        cols = st.columns(7)
        for i in range(7):
            with cols[i]:
                # 合法手でない列のボタンは無効化
                if st.button(f"{i+1}", key=f"col_{i}", disabled=(i not in legal_actions)):
                    # プレイヤーの手を処理
                    _, _, done, _ = st.session_state.game.step(i)
                    
                    if st.session_state.game.have_winner():
                        st.session_state.game_over = True
                        st.session_state.winner = 1  # Human
                    elif done:
                        st.session_state.game_over = True
                        st.session_state.winner = 0  # Draw
                    
                    # ターンを切り替えて再描画
                    st.rerun()

    else: # is_ai_turn
        # --- AIのターン ---
        st.session_state.message = "AIが思考中です..."
        message_placeholder.info(st.session_state.message)

        with st.spinner("AIが思考中..."):
            time.sleep(0.5) # 思考しているように見せるためのウェイト
            ai_action = get_ai_action(st.session_state.game, model, device)
            
            if ai_action is not None:
                _, _, done, _ = st.session_state.game.step(ai_action)
                
                if st.session_state.game.have_winner():
                    st.session_state.game_over = True
                    st.session_state.winner = -1  # AI
                elif done:
                    st.session_state.game_over = True
                    st.session_state.winner = 0  # Draw

        st.session_state.message = "あなたの番です。下のボタンから列を選んでください。"
        st.rerun()
