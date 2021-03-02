# `deep-finance`: Deep Learning for Finance

## [Content](#content)

<table>
<tr><td colspan="2"><a href="#dataset">1. Dataset</a></td></tr> 
<tr><td colspan="2"><a href="#paper">2. Paper</a></td></tr>
<tr>
    <td>&emsp;<a href="#stock-prediction">2.1 Stock Prediction</a></td>
    <td>&ensp;<a href="#portfolio-selection">2.2 Portfolio Selection</a></td>
</tr>
<tr>
    <td>&emsp;<a href="#risk-control">2.3 Risk Control</a></td>
    <td>&ensp;<a href="#finance-nlp">2.4 Finance NLP</a></td>
</tr>
<tr>
    <td>&emsp;<a href="#blockchain">2.5 Blockchain</a></td>
    <td>&ensp;<a href="#market-maker">2.6 Market Maker</a></td>
</tr>
<tr>
    <td>&emsp;<a href="#others">2.7 Others</a></td>
    <td></td>
</tr>
<tr><td colspan="2"><a href="#book">3. Book</a></td></tr>
<tr><td colspan="2"><a href="#disscussion-group">4. Disscussion Group</a></td></tr>
</table>


## [Dataset](#content)
| Dataset                                                                                              | Task                                  | Describe                                                                                                                                                                   |
| ---------------------------------------------------------------------------------------------------- | ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [StockNet](https://github.com/yumoxu/stocknet-dataset)                                               | Stock Movement Prediction             | A comprehensive dataset for stock movement prediction from tweets and historical stock prices.                                                                             |
| [EarningsCall](https://github.com/GeminiLn/EarningsCall_Dataset)                                     | Stock Risk Prediction                 | The earnings conference call dataset of S&P 500 companies.                                                                                                                 |
| [FinSBD-2019](https://sites.google.com/nlg.csie.ntu.edu.tw/finnlp/)                                  | Financial Sentence Boundary Detection | The FinSBD-2019 dataset contains financial text that had been pre-segmented automatically, which can be used for Financial Sentence Boundary Detection.                    |
| [Financial Phrasebank]( https://www.researchgate.net/publication/251231364_FinancialPhraseBank-v10/) | Financial Sentence Boundary Detection | Financial Phrasebank dataset consists of 4845 English sentences selected randomly from financial news found on LexisNexis database.                                        |
| [FiQA ](https://sites.google.com/view/fiqa/home/)                                                    | Financial Question Answering          | Financial QA dataset is built by crawling Stack exchange posts under the Investment topic in the period between 2009 and 2017.                                             |
| [FiQA SA](https://sites.google.com/view/fiqa/home/)                                                  | Financial Sentiment Analysis          | FiQA SA dataset includes two types of discourse: financial news headlines and financial microblogs, with manually annotated target entities, sentiment scores and aspects. |

## [Paper](#content)
### [Stock Prediction](#content)

1. **Applications of deep learning in stock market prediction: recent progress**. arxiv 2020. [paper](https://arxiv.org/abs/2003.01859)
   
   *Weiwei Jiang*

2. **Individualized Indicator for All: Stock-wise Technical Indicator Optimization with Stock Embedding**. KDD 2019. [paper](https://www.kdd.org/kdd2019/accepted-papers/view/individualized-indicator-for-all-stock-wise-technical-indicator-optimizatio)

    *Zhige Li, Derek Yang, Li Zhao, Jiang Bian, Tao Qin and Tie-Yan Liu*

3. **Investment Behaviors Can Tell What Inside: Exploring Stock Intrinsic Properties for Stock Trend Prediction**. KDD 2019. [paper](https://www.kdd.org/kdd2019/accepted-papers/view/investment-behaviors-can-tell-what-inside-exploring-stock-intrinsic-propert)

    *Chi Chen, Li Zhao, Jiang Bian, Chunxiao Xing and Tie-Yan Liu*

4. **Exploring Graph Neural Networks for Stock Market Predictions with Rolling Window Analysis**. CoRR 2019. [paper](https://arxiv.org/abs/1909.10660)

    *Daiki Matsunaga, Toyotaro Suzumura, Toshihiro Takahashi*

5. **Temporal Relational Ranking for Stock Prediction**. TOIS 2019 . [paper](https://arxiv.org/abs/1809.09441)<!-- . [code](https://github.com/fulifeng/Temporal_Relational_Stock_Ranking) -->

    *Fuli Feng, Xiangnan He, Xiang Wang, Cheng Luo, Yiqun Liu, Tat-Seng Chua*

6. **Incorporating Corporation Relationship via Graph Convolutional Neural Networks for Stock Price Prediction**. CIKM 2018 . [paper](https://dl.acm.org/doi/pdf/10.1145/3269206.3269269)

    *Yingmei Chen, Zhongyu Wei, Xuanjing Huang*

7. **Knowledge-Driven Event Embedding for Stock Prediction**. COLING 2016 . [paper](https://www.aclweb.org/anthology/C16-1201)

    *Xiao Ding, Yue Zhang, Ting Liu, Junwen Duan*

8. **HATS: A Hierarchical Graph Attention Network for Stock Movement Prediction**. arxiv 2019 . [paper](https://arxiv.org/abs/1908.07999) <!-- [\[code\]](https://github.com/dmis-lab/hats) -->

    *Raehyun Kim, Chan Ho So, Minbyul Jeong, Sanghoon Lee, Jinkyu Kim, Jaewoo Kang*

9. **Hierarchical Complementary Attention Network for Predicting Stock Price Movements with News** . CIKM 18 . [paper](https://dl.acm.org/doi/pdf/10.1145/3269206.3269286)

    *Qikai Liu, Xiang Cheng, Sen Su, Shuguang Zhu*

10. **Stock Movement Prediction from Tweets and Historical Prices** . ACL 2018 . [paper](https://www.aclweb.org/anthology/P18-1183) <!-- [\[code\]](https://github.com/yumoxu/stocknet-\[code\]) -->

    *Yumo Xu, Shay B. Cohen*

<details><summary> more </summary>

11. **What You Say and How You Say It Matters: Predicting Financial Risk Using Verbal and Vocal Cues** . ACL 2019 . [paper](https://www.aclweb.org/anthology/P19-1038)

    *Yu Qin, Yi Yang*

12. **Listening to Chaotic Whispers: A Deep Learning Framework for News-oriented Stock Trend Prediction** . WSDM 2018 . [paper](https://arxiv.org/abs/1712.02136)

    *Ziniu Hu, Weiqing Liu, Jiang Bian, Xuanzhe Liu*

13. **Enhancing Stock Movement Prediction with Adversarial Training** . IJCAI 2019 . [paper](https://www.ijcai.org/Proceedings/2019/0810.pdf) <!-- [\[code\]](model/adv-alstm.py) -->

    *Fuli Feng, Huimin Chen, Xiangnan He, Ji Ding, Maosong Sun, Tat-Seng Chua*

14. **Multi-task Recurrent Neural Networks and Higher-order Markov Random Fields for Stock Price Movement Prediction** . KDD 2019 . [paper](https://www.kdd.org/kdd2019/accepted-papers/view/multi-task-recurrent-neural-network-and-higher-order-markov-random-fields-f)

    *Chang Li (School of Computer Science University of Sydney);Dongjin Song ( Capital Market CRC);Dacheng Tao (NEC);*

15. **Stock Price Prediction via Discovering Multi-Frequency Trading Patterns** . KDD 2017 . [paper](http://www.eecs.ucf.edu/~gqi/publications/kdd2017_stock.pdf) <!-- [\[code\]](model/sfm.py) -->

    *Liheng Zhang, Charu C. Aggarwal, Guojun Qi*

16. **A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction** . IJCAI 2017 . [paper](https://arxiv.org/abs/1704.02971) <!-- [\[code\]](model/da-rnn.py) -->

    *Yao Qin, Dongjin Song, Haifeng Chen, Wei Cheng, Guofei Jiang, Garrison Cottrell*

17. **Modeling the Stock Relation with Graph Network for Overnight Stock Movement Prediction**. IJCAI 2020: AI in FinTech . [paper](https://www.ijcai.org/Proceedings/2020/0626.pdf)

    *Wei Li, Ruihan Bao, Keiko Harimoto, Deli Chen, Jingjing Xu, Qi Su*

18. **A Quantum-inspired Entropic Kernel for Multiple Financial Time Series Analysis**. IJCAI 2020: AI in FinTech . [paper](https://www.ijcai.org/Proceedings/2020/0614.pdf)

    *Lu Bai, Lixin Cui, Yue Wang, Yuhang Jiao, Edwin R. Hancock*

19. **Hierarchical Multi-Scale Gaussian Transformer for Stock Movement Prediction**. IJCAI 2020: AI in FinTech . [paper](https://www.ijcai.org/Proceedings/2020/0640.pdf)
 
    *Qianggang Ding, Sifan Wu, Hao Sun, Jiadong Guo, Jian Guo*

20. **Multi-scale Two-way Deep Neural Network for Stock Trend Prediction**. IJCAI 2020: AI in FinTech . [paper](https://www.ijcai.org/Proceedings/2020/0628.pdf)

    *Guang Liu, Yuzhao Mao, Qi Sun, Hailong Huang, Weiguo Gao, Xuan Li, Jianping Shen, Ruifan Li, Xiaojie Wang*

</details>


### [Portfolio Selection](#content)
1. **A Two-level Reinforcement Learning Algorithm for Ambiguous Mean-variance Portfolio Selection Problem**. IJCAI 2020: AI in FinTech . [paper](https://www.ijcai.org/Proceedings/2020/0624.pdf)

    *Xin Huang, Duan Li*

2. **Financial Thought Experiment: A GAN-based Approach to Vast Robust Portfolio Selection**. IJCAI 2020: AI in FinTech . [paper](https://www.ijcai.org/Proceedings/2020/0637.pdf)

    *Chi Seng Pun, Lei Wang, Hoi Ying Wong*

3. **MAPS: Multi-Agent reinforcement learning-based Portfolio management System.**. IJCAI 2020: AI in FinTech . [paper](https://www.ijcai.org/Proceedings/2020/0623.pdf)

    *Jinho Lee, Raehyun Kim, Seok-Won Yi, Jaewoo Kang*

4. **Online Portfolio Selection with Cardinality Constraint and Transaction Costs based on Contextual Bandit**. IJCAI 2020: AI in FinTech . [paper](https://www.ijcai.org/Proceedings/2020/0646.pdf)

    *Mengying Zhu, Xiaolin Zheng, Yan Wang, Qianqiao Liang, Wenfang Zhang*

5. **RM-CVaR: Regularized Multiple β-CVaR Portfolio**. IJCAI 2020: AI in FinTech . [paper](https://www.ijcai.org/Proceedings/2020/0629.pdf)

    *Kei Nakagawa, Shuhei Noma, Masaya Abe*

6. **Relation-Aware Transformer for Portfolio Policy Learning**. IJCAI 2020: AI in FinTech . [paper](https://www.ijcai.org/Proceedings/2020/0641.pdf)
   
    *Ke Xu, Yifan Zhang, Deheng Ye, Peilin Zhao, Mingkui Tan*


### [Risk Control](#content)
1. **Financial Risk Analysis for SMEs with Graph-based Supply Chain Mining**. IJCAI 2020: AI in FinTech . [paper](https://www.ijcai.org/Proceedings/2020/0643.pdf)

    *Shuo Yang, Zhiqiang Zhang, Jun Zhou, Yang Wang, Wang Sun, Xingyu Zhong, Yanming Fang, Quan Yu, Yuan Qi*

2. **Federated Meta-Learning for Fraudulent Credit Card Detection**. IJCAI 2020: AI in FinTech . [paper](https://www.ijcai.org/Proceedings/2020/0642.pdf)
    
    *Wenbo Zheng, Lan Yan, Chao Gou, Fei-Yue Wang*

3. **The Behavioral Sign of Account Theft: Realizing Online Payment Fraud Alert**. IJCAI 2020: AI in FinTech . [paper](https://www.ijcai.org/Proceedings/2020/0636.pdf)
    
    *Cheng WANG*

4. **Phishing Scam Detection on Ethereum: Towards Financial Security for Blockchain Ecosystem**. IJCAI 2020: AI in FinTech . [paper](https://www.ijcai.org/Proceedings/2020/0621.pdf)
    
    *Weili Chen, Xiongfeng Guo, Zhiguang Chen, Zibin Zheng, Yutong Lu*

5. **Vector Autoregressive Weighting Reversion Strategy for Online Portfolio Selection**. IJCAI 2020: AI in FinTech . [paper](https://www.ijcai.org/Proceedings/2020/0616.pdf)
    
    *Xia Cai*

6. **An End-to-End Optimal Trade Execution Framework based on Proximal Policy Optimization**. IJCAI 2020: AI in FinTech . [paper](https://www.ijcai.org/Proceedings/2020/0627.pdf)
    
    *Siyu Lin, Peter A. Beling*

7. **Interpretable Multimodal Learning for Intelligent Regulation in Online Payment Systems**. IJCAI 2020: AI in FinTech . [paper](https://www.ijcai.org/Proceedings/2020/0645.pdf)
    
    *Shuoyao Wang, Diwei Zhu*

8. **Risk Guarantee Prediction in Networked-Loans**. IJCAI 2020: AI in FinTech . [paper](https://www.ijcai.org/Proceedings/2020/0618.pdf)
    
    *Dawei Cheng, Xiaoyang Wang, Ying Zhang, Liqing Zhang*

9.  **Risk-Averse Trust Region Optimization for Reward-Volatility Reduction**. IJCAI 2020: AI in FinTech . [paper](https://www.ijcai.org/Proceedings/2020/0632.pdf)
    
    *Lorenzo Bisi, Luca Sabbioni, Edoardo Vittori, Matteo Papini, Marcello Restelli*


### [Finance NLP](#content)
1. **Deep Semantic Compliance Advisor for Unstructured Document Compliance Checking**. IJCAI 2020: AI in FinTech . [paper](https://www.ijcai.org/Proceedings/2020/0613.pdf)

    *Honglei Guo, Bang An, Zhili Guo, Zhong Su*

2. **"The Squawk Bot": Joint Learning of Time Series and Text Data Modalities for Automated Financial Information Filtering**. IJCAI 2020: AI in FinTech . [paper](https://www.ijcai.org/Proceedings/2020/0634.pdf)

    *Xuan-Hong Dang, Syed Yousaf Shah, Petros Zerfos*

3. **A Unified Model for Financial Event Classification, Detection and Summarization**. IJCAI 2020: AI in FinTech . [paper](https://www.ijcai.org/Proceedings/2020/0644.pdf)

    *Quanzhi Li, Qiong Zhang*

4. **F-HMTC: Detecting Financial Events for Investment Decisions Based on Neural Hierarchical Multi-Label Text Classification**. IJCAI 2020: AI in FinTech . paper[](https://www.ijcai.org/Proceedings/2020/0619.pdf)

    *Xin Liang, Dawei Cheng, Fangzhou Yang, Yifeng Luo, Weining Qian, Aoying Zhou*

5. **Financial Risk Prediction with Multi-Round Q&A Attention Network**. IJCAI 2020: AI in FinTech . [paper](https://www.ijcai.org/Proceedings/2020/0631.pdf)

    *Zhen Ye, Yu Qin, Wei Xu*

6. **FinBERT: A Pre-trained Financial Language Representation Model for Financial Text Mining**. IJCAI 2020: AI in FinTech . [paper](https://www.ijcai.org/Proceedings/2020/0622.pdf)

    *Zhuang Liu, Degen Huang, Kaiyu Huang, Zhuang Li, Jun Zhao*

7. **Two-stage Behavior Cloning for Spoken Dialogue System in Debt Collection**. IJCAI 2020: AI in FinTech . [paper](https://www.ijcai.org/Proceedings/2020/0639.pdf)

    *Zihao Wang, Jia Liu, Hengbin Cui, Chunxiang Jin, Minghui Yang, Yafang Wang, Xiaolong Li, Renxin Mao*


### [Blockchain](#content)
1. **BitcoinHeist: Topological Data Analysis for Ransomware Prediction on the Bitcoin Blockchain**. IJCAI 2020: AI in FinTech . [paper](https://www.ijcai.org/Proceedings/2020/0612.pdf)

    *Cuneyt G. Akcora, Yitao Li, Yulia R. Gel, Murat Kantarcioglu*

2. **SEBF: A Single-Chain based Extension Model of Blockchain for Fintech**. IJCAI 2020: AI in FinTech . [paper](https://www.ijcai.org/Proceedings/2020/0620.pdf)

    *Yimu Ji, Weiheng Gu, Fei Chen, Xiaoying Xiao, Jing Sun, Shangdong Liu, Jing He, Yunyao Li, Kaixiang Zhang, Fen Mei, Fei Wu*

3. **Infochain: A Decentralized, Trustless and Transparent Oracle on Blockchain**. IJCAI 2020: AI in FinTech . [paper](https://www.ijcai.org/Proceedings/2020/0635.pdf)

    *Naman Goel, Cyril van Schreven, Aris Filos-Ratsikas, Boi Faltings*


### [Market Maker](#content)
1. **Market Manipulation: An Adversarial Learning Framework for Detection and Evasion**. IJCAI 2020: AI in FinTech . [paper](https://www.ijcai.org/Proceedings/2020/0638.pdf)

    *Xintong Wang, Michael P. Wellman*

2. **Data-Driven Market-Making via Model-Free Learning**. IJCAI 2020: AI in FinTech . [paper](https://www.ijcai.org/Proceedings/2020/0615.pdf)

    *Yueyang Zhong, YeeMan Bergstrom, Amy Ward*

3. **Robust Market Making via Adversarial Reinforcement Learning**. IJCAI 2020: AI in FinTech . [paper](https://www.ijcai.org/Proceedings/2020/0633.pdf)

    *Thomas Spooner, Rahul Savani*


### [Others](#content)
1. **IGNITE: A Minimax Game Toward Learning Individual Treatment Effects from Networked Observational Data**. IJCAI 2020: AI in FinTech . [paper](https://www.ijcai.org/Proceedings/2020/0625.pdf)

    *Ruocheng Guo, Jundong Li, Yichuan Li, K. Selçuk Candan, Adrienne Raglin, Huan Liu*

2. **Task-Based Learning via Task-Oriented Prediction Network with Applications in Finance**. IJCAI 2020: AI in FinTech . [paper](https://arxiv.org/abs/1910.09357)

    *Di Chen, Yada Zhu, Xiaodong Cui, Carla P. Gomes*

3. **WATTNet: Learning to Trade FX via Hierarchical Spatio-Temporal Representation of Highly Multivariate Time Series**. IJCAI 2020: AI in FinTech . [paper](https://www.ijcai.org/Proceedings/2020/0630.pdf)

    *Michael Poli, Jinkyoo Park, Ilija Ilievski*


## [Book](#content)
1. [The Econometrics of Financial Markets](https://item.jd.com/1107212917.html)

    *John Y. Campbell, Andrew W. Lo, A. Craig Mackinlay*

2. [Advances in Financial Machine Learning](https://item.jd.com/39205783211.html)

    *Marcos Lopez de Prado*

3. [Financial Decisions and Markets: A Course in Asset Pricing](https://www.semanticscholar.org/paper/Financial-Decisions-and-Markets%3A-A-Course-in-Asset-Campbell/f413566883c4be4f8e55e275e3f70b2aebf9e8fc)

    *J. Campbell*

## [Disscussion Group](#content)
对于AI+Finance方向感兴趣的童鞋，欢迎扫描下面的二维码学习交流：

<html>
    <table style="margin-left: 20px; margin-right: auto;">
        <tr>
            <td>
                <img src="./docs/figs/Zhihu.png" />
            </td>
            <td>
                <img src="./docs/figs/OA.png" />
            </td>
            <td>
                <img src="./docs/figs/Wechat.png">
            </td>
        </tr>
    </table>
</html>