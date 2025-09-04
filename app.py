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
    html = f'''
    <div class="board-wrap">
        <span class="board">{header}<br/>{board_body}</span>
    </div>
    '''
    st.markdown(html, unsafe_allow_html=True)


# --- Streamlit アプリケーションのメイン処理 ---

st.title("Connect4（pytorch版）")

# レスポンシブCSS（スマホ最適化）
st.markdown(
        """
            <style>
            .board-wrap { display: flex; justify-content: center; }
            .board { font-family: 'Segoe UI Symbol','Noto Sans Symbols 2','Apple Color Emoji','MS Gothic','Osaka-Mono',monospace; font-size: 22px; line-height: 1.05; letter-spacing: 1px; }
            @media (max-width: 600px) {
                .board { font-size: 18px; line-height: 1.0; letter-spacing: 0.5px; }
                .stButton>button { padding: 0.4rem 0.6rem; }
            }
            </style>
        """,
        unsafe_allow_html=True,
)

model, device = load_model()

# セッション初期化（app_v2.pyの体裁に準拠）
if 'game_started' not in st.session_state:
    st.session_state.game_started = False
if 'first_player' not in st.session_state:
    st.session_state.first_player = None
if 'game' not in st.session_state:
    st.session_state.game = Connect4()
if 'game_over' not in st.session_state:
    st.session_state.game_over = False
if 'winner' not in st.session_state:
    st.session_state.winner = None
if 'message' not in st.session_state:
    st.session_state.message = "先手/後手を選んでください。"
if 'ai_scores' not in st.session_state:
    st.session_state.ai_scores = None

def start_new_game(first_player: str):
    st.session_state.game = Connect4()
    # 先手: human -> env.player=1, 後手: ai -> env.player=-1（AIが先に打つ）
    if first_player == 'ai':
        st.session_state.game.player = -1
    else:
        st.session_state.game.player = 1
    st.session_state.game_over = False
    st.session_state.winner = None
    st.session_state.message = (
        "あなたの番です。下のボタンから列を選んでください。" if first_player == 'human' else "AIが先手です。"
    )
    st.session_state.ai_scores = None
    st.session_state.game_started = True
    st.session_state.first_player = first_player

def render_action_picker(legal_actions):
    """横スクロール不要・省スペースな水平ラジオと『置く』ボタンの組み合わせ。"""
    if not legal_actions:
        return None
    opts = [i + 1 for i in legal_actions]
    # 水平ラジオ（モバイルでは複数行に自然に折り返す）
    selected = st.radio("列を選択", options=opts, horizontal=True, key="action_radio")
    # 実行ボタンを右に配置（レイアウトの都合で列に分割）
    c1, c2 = st.columns([3, 1])
    with c2:
        go = st.button("置く", key="place_button")
    if go and selected is not None:
        return int(selected) - 1
    return None

# ゲーム開始前：先手/後手選択
if not st.session_state.game_started:
    st.info("先手（人間）か後手（AI）を選んでください。")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("先手（人間）で始める", key="start_human"):
            start_new_game('human')
            st.rerun()
    with col2:
        if st.button("後手（AI）で始める", key="start_ai"):
            start_new_game('ai')
            st.rerun()
    st.stop()

# ゲーム進行中：盤面/メッセージ
display_board(st.session_state.game.board)
message_placeholder = st.empty()
message_placeholder.info(st.session_state.message)

# AIの評価表示（任意）
if st.session_state.ai_scores:
    with st.expander("AIの各列の評価値"):
        scores = st.session_state.ai_scores
        cols = st.columns(7)
        for i in range(7):
            with cols[i]:
                val = scores.get(i, None)
                if val is None:
                    st.write(f"{i+1}: -")
                else:
                    st.write(f"{i+1}: {val:.3f}")

# 終了後のリザルト
if st.session_state.game_over:
    if st.session_state.winner == 1:
        st.success("🎉 あなたの勝ちです！おめでとうございます！")
    elif st.session_state.winner == -1:
        st.error("🤖 AIの勝ちです。")
    elif st.session_state.winner == 0:
        st.warning("引き分けです。")
    else:
        st.info("ゲーム終了")

    if st.button("新しいゲームを始める"):
        st.session_state.game_started = False
        st.session_state.first_player = None
        st.rerun()

else:
    current_player = st.session_state.game.get_player()

    if current_player == 1:  # 人間のターン
        legal_actions = st.session_state.game.legal_actions()
        action = render_action_picker(legal_actions)
        if action is not None:
            _, reward, done = st.session_state.game.step(action)
            if done:
                st.session_state.game_over = True
                if reward == 1:
                    st.session_state.winner = 1  # Human勝利
                elif len(st.session_state.game.legal_actions()) == 0:
                    st.session_state.winner = 0  # 引き分け
                else:
                    st.session_state.winner = None
            st.session_state.ai_scores = None
            st.rerun()

    else:  # AIのターン
        st.session_state.message = "AIが思考中です..."
        message_placeholder.info(st.session_state.message)
        with st.spinner("AIが思考中..."):
            time.sleep(1.0)
            ai_action, scores = get_ai_action_and_scores(st.session_state.game, model, device)
            st.session_state.ai_scores = scores
            if ai_action is not None:
                _, reward, done = st.session_state.game.step(ai_action)
                if done:
                    st.session_state.game_over = True
                    if reward == 1:
                        st.session_state.winner = -1  # AI勝利
                    elif len(st.session_state.game.legal_actions()) == 0:
                        st.session_state.winner = 0  # 引き分け
                    else:
                        st.session_state.winner = None
        st.session_state.message = "あなたの番です。下のボタンから列を選んでください。"
        st.rerun()
#
#4層のニューラルネットワークを自己対戦で強化学習(32回の対戦を80万回＝2560万回)
#損失関数: MSE(平均二乗誤差)
#最適化関数: Adadelta(勾配と更新量の移動平均を使い、学習率を自動計算する) 学習率: 0.1
#
#＜ニューラルネット＞
#出力層    1 = 推奨度(報酬) 勝ち:1.0, 同点:0.5, 負け:-1.5 (割引率:0.99)
#中間層  364
#中間層  182
#入力層　 91 = 盤面状態(7列 × 6段 × 2人) ＋ アクション(7列)
#　　　　　　　 コイン(なし:0, あり:1)　　　 非選択:0, 選択:1
#パラメータ数 83721 = (91*182+182) + (182*364+364) + (364*1+1)
