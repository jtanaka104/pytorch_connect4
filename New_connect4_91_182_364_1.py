import numpy
import random
import torch
import torch.nn as nn
import datetime
import pathlib

useGPU = 1

#####################################################################################################
# Connect4(環境)
#####################################################################################################
class Connect4:
    #################################################################################################
    # 初期化
    #################################################################################################
    def __init__(self):
        # ボード作成と初期化
        self.board = numpy.zeros((6, 7), dtype="int32")
        # 「1:先手」で初期化、「0:後手」
        self.player = 1

    #################################################################################################
    # 先手・後手の取得
    #################################################################################################
    def get_player(self):
        return self.player

    #################################################################################################
    # リセット
    # 盤面の状態
    #################################################################################################
    def reset(self):
        # ボード作成と初期化
        self.board = numpy.zeros((6, 7), dtype="int32")
        # 「1:先手」で初期化、「-1:後手」
        self.player = 1
        return self.get_observation()

    #################################################################################################
    # AIが選択したActionを実行(action: 0 - 7)
    # 盤面の状態、報酬、勝利状態か否か
    #################################################################################################
    def step(self, action):
        for i in range(6):
            if self.board[i][action] == 0:
                self.board[i][action] = self.player
                break

        done = self.have_winner() or len(self.legal_actions()) == 0

        reward = 1 if self.have_winner() else 0

        self.player *= -1

        return self.get_observation(), reward, done

    #################################################################################################
    # 盤面の状態を返却する
    # [ 自石の場所, 他石の場所]
    #################################################################################################
    def get_observation(self):
        board_player1 = numpy.where(self.board == 1, 1, 0)
        board_player2 = numpy.where(self.board == -1, 1, 0)
        return numpy.array([board_player1, board_player2], dtype="int32").flatten()

    #################################################################################################
    # 合法手を返却する
    #################################################################################################
    def legal_actions(self):
        legal = []
        for i in range(7):
            if self.board[5][i] == 0:
                legal.append(i)
        return legal

    #################################################################################################
    # 打ち手が勝利状態か確認する
    #################################################################################################
    def have_winner(self):
        # Horizontal check
        for i in range(4):
            for j in range(6):
                if (
                    self.board[j][i] == self.player
                    and self.board[j][i + 1] == self.player
                    and self.board[j][i + 2] == self.player
                    and self.board[j][i + 3] == self.player
                ):
                    return True

        # Vertical check
        for i in range(7):
            for j in range(3):
                if (
                    self.board[j][i] == self.player
                    and self.board[j + 1][i] == self.player
                    and self.board[j + 2][i] == self.player
                    and self.board[j + 3][i] == self.player
                ):
                    return True

        # Positive diagonal check
        for i in range(4):
            for j in range(3):
                if (
                    self.board[j][i] == self.player
                    and self.board[j + 1][i + 1] == self.player
                    and self.board[j + 2][i + 2] == self.player
                    and self.board[j + 3][i + 3] == self.player
                ):
                    return True

        # Negative diagonal check
        for i in range(4):
            for j in range(3, 6):
                if (
                    self.board[j][i] == self.player
                    and self.board[j - 1][i + 1] == self.player
                    and self.board[j - 2][i + 2] == self.player
                    and self.board[j - 3][i + 3] == self.player
                ):
                    return True

        return False

    #################################################################################################
    # エキスパートのエージェント
    #################################################################################################
    def expert_action(self):
        board = self.board
        action = numpy.random.choice(self.legal_actions())
        for k in range(3):
            for l in range(4):
                sub_board = board[k : k + 4, l : l + 4]
                # Horizontal and vertical checks
                for i in range(4):
                    if abs(sum(sub_board[i, :])) == 3:
                        ind = numpy.where(sub_board[i, :] == 0)[0][0]
                        if numpy.count_nonzero(board[:, ind + l]) == i + k:
                            action = ind + l
                            if self.player * sum(sub_board[i, :]) > 0:
                                return action

                    if abs(sum(sub_board[:, i])) == 3:
                        action = i + l
                        if self.player * sum(sub_board[:, i]) > 0:
                            return action
                # Diagonal checks
                diag = sub_board.diagonal()
                anti_diag = numpy.fliplr(sub_board).diagonal()
                if abs(sum(diag)) == 3:
                    ind = numpy.where(diag == 0)[0][0]
                    if numpy.count_nonzero(board[:, ind + l]) == ind + k:
                        action = ind + l
                        if self.player * sum(diag) > 0:
                            return action

                if abs(sum(anti_diag)) == 3:
                    ind = numpy.where(anti_diag == 0)[0][0]
                    if numpy.count_nonzero(board[:, 3 - ind + l]) == ind + k:
                        action = 3 - ind + l
                        if self.player * sum(anti_diag) > 0:
                            return action

        return action

    #################################################################################################
    # 状態の出力
    #################################################################################################
    def render(self, predict):
        wp = predict
        actions = self.legal_actions()
        for action in range(7):
            if action not in actions:
                wp[0][action] = 0.0
        print('       推奨度')
        print((wp[0]*100).astype(int))
        print('      盤面の状態')
        print('[ 0  1  2  3  4  5  6]')
        print([2 if value==-1 else value for value in self.board[5]])
        print([2 if value==-1 else value for value in self.board[4]])
        print([2 if value==-1 else value for value in self.board[3]])
        print([2 if value==-1 else value for value in self.board[2]])
        print([2 if value==-1 else value for value in self.board[1]])
        print([2 if value==-1 else value for value in self.board[0]])
        print()

#####################################################################################################
# 決定
#####################################################################################################
class Decision:
    def __init__(self, states, action):
        self.states  = states
        self.action = action

#####################################################################################################
# アクション選択関数の実装
#####################################################################################################
def get_action(a, r):
    wk = (a * 100).astype(int)
    wk -= min(wk)
    wk += 1
    wk **= 2
    legal = env.legal_actions()
    for k in range(7):
        if k not in legal:
            wk[k] = 0.0
    total = sum(wk)
    fwk = wk / total
    sum_wk = 0.0
    action = 0
    for i in range(7):
        sum_wk += fwk[i]
        if r < sum_wk:
            action = i
            break
    return action, fwk[action]

#####################################################################################################
# ミニバッチ・データ作成 (入力情報と教師情報を返却する)
#####################################################################################################
def create_batch(rate):
    X = numpy.empty((0,91))
    T = numpy.empty((0,1))
    AI = 0
    RND = 0
    STEPS = 0
    AT = 0
    SUM_SMAX = 0

    # ランダム対戦でMB_TIMES回プレイアウトする
    for tim in range(MB_TIMES):
        # 状態とアクションのリスト
        P1 = numpy.empty(0)
        P2 = numpy.empty(0)
        # ゲームをリセット
        states = env.reset()
        # プレイアウトするまでループ
        player = 1
        bonus = 0.5
        while True:
            # STEPをカウント
            STEPS += 1
            bonus -= 0.01
            
            # 手番を設定
            player = env.get_player()
            
            # 合法手を取得
            actions = env.legal_actions()
                
            # 乱数を取得
            rnd_num = random.random()
            
            # AIによる各Actionの推奨度を取得
            predicts = predict()

            # 推奨度に応じた割合で選択
            if (rate < 0):
                action, ratio = get_action(predicts, rnd_num)
                SUM_SMAX += ratio
                AT += 1
            # ランダムの選択
            elif (rnd_num < rate):
                action = random.choice(actions)
                RND += 1
            # 最善手の選択
            else:
                # 合法手の中で最も期待値の大きなActionを抽出
                max_predict = -999999
                action = 0
                for i in actions:
                    if max_predict < predicts[i]:
                        max_predict = predicts[i]
                        action = i
                AI += 1
            # 状態とアクションをリストに追加
            decision = Decision(states, action)
            if 0 < player:
                P1 = numpy.append(P1, decision)
            else:
                P2 = numpy.append(P2, decision)
            # 選択したランダムな合法手を実行
            states, reward, done = env.step(action)
            # 終了ならばループを抜ける
            if done:
                # 期待値定義
                E1 = 0.0
                E2 = 0.0
                # 勝敗がついた場合
                if reward == 1:
                    # 勝ったのが先手の場合
                    if player == 1:
                        E1 = 1.0 + bonus
                        E2 = -1.5 - bonus
                    # 勝ったのが後手の場合
                    else:
                        E1 = -1.5 - bonus
                        E2 = 1.0 + bonus
                # 勝敗がつかなかった場合
                else:
                        E1 = 0.5
                        E2 = 0.5
                # 先手の入力値(状態＋手)と教師データ(期待値)を作成
                for index in range(len(P1)):
                    ws = P1[index].states
                    wa = [0., 0., 0., 0., 0., 0., 0.]
                    wa[P1[index].action] = 1.0
                    wx = numpy.append(ws, wa)
                    X = numpy.append(X, [wx], axis=0)
                    T = numpy.append(T, [[E1]], axis=0)
                # 後手の入力値(状態＋手)と教師データ(期待値)を作成
                for index in range(len(P2)):
                    ws = P2[index].states
                    wa = [0., 0., 0., 0., 0., 0., 0.]
                    wa[P2[index].action] = 1.0
                    wx = numpy.append(ws, wa)
                    X = numpy.append(X, [wx], axis=0)
                    T = numpy.append(T, [[E2]], axis=0)
                break
    return X, T, AI, RND, STEPS, AT, SUM_SMAX

#####################################################################################################
# 訓練
#####################################################################################################
def train(rate):
    # グローバル変数使用宣言
    global epoch
    
    # 操作説明
    print('訓練を中断したい場合はCTL-Cを押下して下さい')
    
    # 例外検出区間
    try:
        # ミニバッチをCTL-Cが押下されるまで実施する
        AI_CNT = 0
        RND_CNT = 0
        STEPS_CNT = 0
        GAMES_CNT = 0
        AT_CNT = 0
        SMAX_CNT = 0
        while True:
            # ミニバッチ・データ作成を実行し、入力情報と教師情報を取得する
            X, T, AI, RND, STEPS, AT, SUM_SMAX = create_batch(rate)
            AI_CNT += AI
            RND_CNT += RND
            STEPS_CNT += STEPS
            GAMES_CNT += MB_TIMES
            AT_CNT += AT
            SMAX_CNT += SUM_SMAX
            
            # 勾配を0に初期化する必要がある
            optimizer.zero_grad()

            # 順伝播
            tX = torch.FloatTensor(X)
            if useGPU:
                tX = tX.cuda()
            Y = model.forward(tX)

            # 損失を計算
            tT = torch.FloatTensor(T)
            if useGPU:
                tT = tT.cuda()
            loss = loss_fn(Y, tT)

            # 逆伝播
            loss.backward()

            # オプティマイザがパラメータの更新量を計算し、モデルに返してパラメータ更新
            optimizer.step()

            # RP_FREQ エポック毎に損失の値を表示
            if (epoch+1) % RP_FREQ == 0:
                print("epoch: %d  loss: %f  AI: %d  RND: %d  AT: %d AVR: %d RAT: %f" % (epoch+1 ,float(loss), AI_CNT, RND_CNT, AT_CNT, STEPS_CNT/GAMES_CNT, SMAX_CNT/AT_CNT))
                
                # 盤面をリセット
                env.reset()
                
                # AIによる各Actionの推奨度を取得
                predicts = predict()
                
                # 盤面の表示
                env.render(predicts.reshape(1,7))
                
                # カウンタ・リセット
                AI_CNT = 0
                RND_CNT = 0
                STEPS_CNT = 0
                GAMES_CNT = 0
                AT_CNT = 0
                SMAX_CNT = 0
            epoch += 1
    # 例外発生
    except:
        pass

#################################################################################################
# 人間にアクションを選択させます。
#################################################################################################
def human_to_action():
    while True:
        digits = int(
            input(
                f"Enter digits(0 - 6): "
            )
        )
        if digits == 9:
            choice = -1
            break
        else:
            choice = digits
            if (
                choice in env.legal_actions()
                and 0 <= choice
                and choice <= 6
            ):
                break
        print("Wrong input, try again")
    return choice

#####################################################################################################
# AIによる各Actionの推奨度を取得
#####################################################################################################
def predict():
    # 盤面の状態を取得
    states = env.get_observation()
    
    # 全Actionの入力データ作成
    X = numpy.empty((0,91))
    for action in range(7):
        wa = [0., 0., 0., 0., 0., 0., 0.]
        wa[action] = 1.0
        wx = numpy.append(states, wa)
        X = numpy.append(X, [wx], axis=0)
    
    # 勾配を0に初期化する必要がある
    optimizer.zero_grad()
    
    # 順伝播
    fT = torch.FloatTensor(X)
    if useGPU:
        fT = fT.cuda()
    Y = model.forward(fT)
    Z = Y.to('cpu').detach().numpy().copy()
    L = Z.flatten()
    
    return L

#####################################################################################################
# 対戦
#####################################################################################################
def play(turn):
    # 操作説明
    print('対戦を途中で終了させたい場合は9を選択して下さい')
    print()
    
    # 盤面をリセット
    env.reset()
    
    # AIによる各Actionの推奨度を取得
    predicts = predict()
    
    # 盤面の表示
    env.render(predicts.reshape(1,7))
    
    while True:
        # [人間が先手 かつ 先手番] or [人間が後手 かつ 後手番]
        if (turn == 1 and 0 < env.get_player()) or (turn == 2 and env.get_player() < 0):
            action = human_to_action()
            if action == -1:
                break
        else:
            # 合法手を取得
            actions = env.legal_actions()
            
            # 合法手の中で最も期待値の大きなActionを抽出
            max_predict = -999999
            action = 0
            for i in actions:
                if max_predict < predicts[i]:
                    max_predict = predicts[i]
                    action = i
            key = ['0', '1', '2', '3', '4', '5', '6']
            print('AI selected (0 - 6):', key[action])
        
        # 選択した手を実行
        states, reward, done = env.step(action)
        
        # AIのAction候補を取得
        predicts = predict()
        
        # 盤面の表示
        env.render(predicts.reshape(1,7))
        
        if done:
            break

#####################################################################################################
# クラス定義(ニューラル・ネットワーク)
#####################################################################################################
# ニューラル・ネットワーク (91→182→364→1)
#####################################################################################################
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1   = nn.Linear(91,182)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Linear(182,364)
        self.relu2 = nn.ReLU()
        self.fc3   = nn.Linear(364,1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

#####################################################################################################
# Save
#####################################################################################################
def save():
    results_path = pathlib.Path("results") / pathlib.Path(__file__).stem / datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    results_path.mkdir(parents=True, exist_ok=True)
    data = {
               'epoch': epoch,
               'model': model.state_dict(),
               'optim': optimizer.state_dict()
           };
    torch.save(data, results_path / "model.cpt")

#####################################################################################################
# Load
#####################################################################################################
def load():
    # グローバル変数使用宣言
    global model
    global optimizer
    global epoch
    
    # Configure running options
    options = ["Specify paths manually"] + sorted(
        (pathlib.Path("results") / pathlib.Path(__file__).stem).glob("*/")
    )
    options.reverse()
    print()
    for i in range(len(options)):
        print(f"{i}. {options[i]}")
    choice = input("Enter a number to choose a model to load: ")
    valid_inputs = [str(i) for i in range(len(options))]
    while choice not in valid_inputs:
        choice = input("Invalid input, enter a number listed above: ")
    choice = int(choice)

    if choice == (len(options) - 1):
        # manual path option
        results_path = input(
            "Enter a path to the model.cpt, or ENTER if none: "
        )
        while results_path and not pathlib.Path(results_path).is_file():
            results_path = input("Invalid checkpoint path. Try again: ")
        replay_buffer_path = input(
            "Enter a path to the replay_buffer.pkl, or ENTER if none: "
        )
        while replay_buffer_path and not pathlib.Path(replay_buffer_path).is_file():
            replay_buffer_path = input("Invalid replay buffer path. Try again: ")
    else:
        results_path = options[choice] / "model.cpt"

    data = torch.load(results_path)
    model.load_state_dict(data['model'])
    optimizer.load_state_dict(data['optim'])
    epoch = data['epoch']

#####################################################################################################
# メイン
#####################################################################################################
# Connect4環境を作成
env = Connect4()

# ニューラル・ネットワーク 
model = Net()
if useGPU:
    model = model.cuda()

# 損失関数
loss_fn = nn.MSELoss()

# 最適化
#optimizer = torch.optim.SGD(params = model.parameters() ,      lr = 0.01)
#optimizer = torch.optim.Adagrad(params = model.parameters() ,  lr = 0.1)
#optimizer = torch.optim.RMSprop(params = model.parameters() ,  lr = 0.1)
optimizer = torch.optim.Adadelta(params = model.parameters() , lr = 0.1)
#optimizer = torch.optim.Adam(params = model.parameters() ,      lr = 0.1)
#optimizer = torch.optim.AdamW(params = model.parameters() ,    lr = 0.1)

# EPOCH
epoch = 0

# 設定項目
RP_FREQ = 100    # 報告頻度
MB_TIMES = 32    # ミニバッチ回数

# メニューループ
while True:
    # Configure running options
    print()
    print('### メニュー ###')
    options = [
        "[t]une parameter",
        "[d]isplay parameter",
        "[a]bility rate train",
        "[r]andom train",
        "[f]ixed rate train",
        "[0]th play",
        "[1]st play",
        "[2]nd play",
        "[s]ave model",
        "[l]oad model",
        "[e]xit",
    ]
    for i in range(len(options)):
        print(f"{options[i]}")
        
    choice = input("Choose an action: ")
    
    if (choice == "t"):
        tune()
    elif (choice == "d"):
        disp()
    elif (choice == "a"):
        train(-1)
    elif (choice == "r"):
        train(1)
    elif (choice == "f"):
        rate = input("Exploration rate (1-99): ")
        train(float(rate)/100)
    elif (choice == "0"):
        play(0)
    elif (choice == "1"):
        play(1)
    elif (choice == "2"):
        play(2)
    elif (choice == "s"):
        save()
    elif (choice == "l"):
        load()
    elif (choice == "e"):
        break
