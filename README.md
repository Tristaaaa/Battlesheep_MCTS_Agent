## Battlesheep_MCTS_Agent
此 game project 需要使用某種自訂的遊戲策略，在 Battle sheep 這個遊戲中取得勝利，也就是要與其他羊群競爭，想辦法取得更多地盤。
而實驗最終採取的遊戲策略是利用 Monte Carlo Tree Search(MCTS) 實作出 game agent，以下將逐一敘述 MCTS 的實作細節以及實驗結果。

### How game AI works
利用 MCTS 來做 tree search，是因為在模擬對戰的過程中，計算的過程是以樹狀結構紀錄，從 root node 往下展開後，根據 child node 的對戰結果，
往回 backup 給中途的每個 node 來計算勝率。以 battle sheep 這個遊戲來說，若真的去計算每一個 node 會讓整個樹狀結構太大，而且也會超過時間限制，
因此利用 MCTS 來減小 search space，同時又能夠搜尋最大勝率的 node。

### Algorithm
- MCTS主要分成 selection、expansion、simulation、backpropagation 四大步驟。
![image](https://github.com/Tristaaaa/Battlesheep_MCTS_Agent/blob/main/test/mcts_own.png)img source: https://www.geeksforgeeks.org/ml-monte-carlo-tree-search-mcts/
1. Selection: 選擇要進行擴展的 child node。
2. Expansion: 展開下一層。
3. Simulation(rollout): 計算這個 node 的 value(估計勝算)。
4. Backpropagation: 由下往上更新各個 node 自己的勝率。

### Implementation(functions explanation)
以下將逐一介紹 MCTS agent 中三個重要的 class 以及其 function 的作用。
- MCTS 的 data structure: 
1. root_mapstate/ root_sheepstate: 首先，每個 node 都必須儲存當下的棋盤狀態。
2. num_rollouts: 這個 node 總共進行幾次對戰。
3. player: 這個 node 是由哪一方在下棋。
4. 最後，其中 node 的 DS 包括儲存其擁有的 child 個數、自己所指向的 parent node 的 pointer，來幫助 backpropagation 的進行。

- MCTSMeta: store constant parameters，方便進行experiment，尋找best parameters的時候更改參數，比較結果。
- Node:在 search tree 上頭每個 node 都代表各自的 game state。
- UctMctsAgent: Handles MCTS 的整個樹狀結構。
1. Selection: 針對 selection 的部分，採用 UCT(Upper Confidence Bounds To Trees)當作 utility function，
這個 formula 可以讓 MCTS 在平均較高勝率的 node 間移動，同時探索少數可能 node之間的平衡。將 node 帶入 UCT formula後，選出數值最高的 node 進行下個步驟的 expansion。
2. Expansion: 在上一個步驟 selection 選出某個 node 之後，針對這個 node 去擴展它的 children。在這個遊戲中，適當的防守和進攻是相輔相成的，
如果策略集中在進攻模式的話，我方很快就會被敵方擋住去路，反而言之若只是拼命的防守，就有可能會忽略敵方的勝利(特別厲害的對手/ first place)，因此為了兼顧進攻與防守，
會探索所有的 legal move。
3. Simulation(roll_out): 完成前兩步驟後，game AI 會針對所有已經 expand 出來的 node 進行simulation。MCTS agent 會模擬我方和其他玩家輪流隨機對戰，
一直持續直到分出勝負，結束後得到的結果代表這個 node 好壞的估值，類似 evaluation value。雖然 time limit 是設定4秒，但是根據 Monte Carlo 方法的精神：隨機取樣，
只要以這個 node 為 root node 的 child node 總模擬次數夠大，就可以趨近真正的值。
4. Backpropagation: 持續前面 simulation 的步驟一直到遊戲結束，並將勝負結果作為 reward回傳，往 search tree 的 root node 逐一更新。
不斷重複進行 selection、expansion、simulation、backpropagation 四個步驟，直到執行時間超4秒時，就將最佳的 node 輸出給 agent。

### Experiments and Results
- Experiment 1: Modify parameter
> goal: find best exploration parameter for MCTS agent
> 一開始 exploration parameter 為 0(但不能賦0，會出錯)，接著 exploration 比重逐漸增加。因為要兼顧 exploration 與 exploitation，最終選擇 exploration = 0.5，得到不錯的 reward。


- Experiment 2: init_pos 相鄰空格數量所考量的層數
> goal: find best init_pos for MCTS agent
> 嘗試將檢查的層數增加為 2(搜尋的相鄰格數增加為 18 格)，因為自由度提高，所以結果也是滿意的(不容易被擋住)。

- Experiment 3: reward的賦值
> goal: assign optimal reward to get better performance
> MCTS agent 中 node 的 value 是以 reward 去計算勝率的，因此好的reward重要到可以帶領game agent拿第一。






