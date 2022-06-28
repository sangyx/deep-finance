# `deep-finance`: Deep Learning for Finance

> This repository is no longer updated since there are no worthwhile studies. If you are really interested in Deep Learning & Finance, I recommend you to read high quality papers on **Time Series**, **Natural Language Process**, **Graph Neural Networks** and **Finance**.

## [Content](#content)

<table>
<tr><td colspan="2"><a href="#dataset">1. Dataset</a></td></tr> 
<tr><td colspan="2"><a href="#paper">2. Paper</a></td></tr>
<tr>
    <td>&emsp;<a href="#stock-prediction">2.1 Stock Prediction</a></td>
    <td>&ensp;<a href="#portfolio-selection">2.2 Portfolio Selection</a></td>
</tr>
<tr>
    <td>&emsp;<a href="#risk-management">2.3 Risk Management</a></td>
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

7. **Vector Autoregressive Weighting Reversion Strategy for Online Portfolio Selection**. IJCAI 2020: AI in FinTech . [paper](https://www.ijcai.org/Proceedings/2020/0616.pdf)
    
    *Xia Cai*

8. **An End-to-End Optimal Trade Execution Framework based on Proximal Policy Optimization**. IJCAI 2020: AI in FinTech . [paper](https://www.ijcai.org/Proceedings/2020/0627.pdf)
    
    *Siyu Lin, Peter A. Beling*


### [Risk Management](#content)
1. **Financial Risk Analysis for SMEs with Graph-based Supply Chain Mining**. IJCAI 2020: AI in FinTech . [paper](https://www.ijcai.org/Proceedings/2020/0643.pdf)

    *Shuo Yang, Zhiqiang Zhang, Jun Zhou, Yang Wang, Wang Sun, Xingyu Zhong, Yanming Fang, Quan Yu, Yuan Qi*

2. **Federated Meta-Learning for Fraudulent Credit Card Detection**. IJCAI 2020: AI in FinTech . [paper](https://www.ijcai.org/Proceedings/2020/0642.pdf)
    
    *Wenbo Zheng, Lan Yan, Chao Gou, Fei-Yue Wang*

3. **The Behavioral Sign of Account Theft: Realizing Online Payment Fraud Alert**. IJCAI 2020: AI in FinTech . [paper](https://www.ijcai.org/Proceedings/2020/0636.pdf)
    
    *Cheng WANG*

4. **Phishing Scam Detection on Ethereum: Towards Financial Security for Blockchain Ecosystem**. IJCAI 2020: AI in FinTech . [paper](https://www.ijcai.org/Proceedings/2020/0621.pdf)
    
    *Weili Chen, Xiongfeng Guo, Zhiguang Chen, Zibin Zheng, Yutong Lu*

5. **Interpretable Multimodal Learning for Intelligent Regulation in Online Payment Systems**. IJCAI 2020: AI in FinTech . [paper](https://www.ijcai.org/Proceedings/2020/0645.pdf)
    
    *Shuoyao Wang, Diwei Zhu*

6. **Risk Guarantee Prediction in Networked-Loans**. IJCAI 2020: AI in FinTech . [paper](https://www.ijcai.org/Proceedings/2020/0618.pdf)
    
    *Dawei Cheng, Xiaoyang Wang, Ying Zhang, Liqing Zhang*

7.  **Risk-Averse Trust Region Optimization for Reward-Volatility Reduction**. IJCAI 2020: AI in FinTech . [paper](https://www.ijcai.org/Proceedings/2020/0632.pdf)
    
    *Lorenzo Bisi, Luca Sabbioni, Edoardo Vittori, Matteo Papini, Marcello Restelli*

8. **Spotlighting Anomalies using Frequent Patterns**. KDD 2017: Anomaly Detection in Finance . [paper](http://proceedings.mlr.press/v71/kuchar18a/kuchar18a.pdf)
    
    *Jaroslav Kuchař, Vojtěch Svátek*
    
9. **Collective Fraud Detection Capturing Inter-Transaction Dependency**. KDD 2017: Anomaly Detection in Finance . [paper](http://proceedings.mlr.press/v71/cao18a/cao18a.pdf)
    
    *Bokai Cao, Mia Mao, Siim Viidu, Philip S. Yu*
    
10. **Automated System for Data Attribute Anomaly Detection**. KDD 2017: Anomaly Detection in Finance . [paper](http://proceedings.mlr.press/v71/love18a/love18a.pdf)
    
    *Nalin Aggarwal, Alexander Statnikov, Chao Yuan*

<details><summary> more </summary>

11. **Sleuthing for adverse outcomes using anomaly detection**. KDD 2017: Anomaly Detection in Finance . [paper](http://proceedings.mlr.press/v71/miller18a/miller18a.pdf)
    
    *Michelle Miller, Robert Cezeaux*
    
12. **Anomaly detection with density estimation trees**. KDD 2017: Anomaly Detection in Finance . [paper](http://proceedings.mlr.press/v71/ram18a/ram18a.pdf)
    
    *Parikshit Ram, Alexander Gray*
    
13. **Binned Kernels for Anomaly Detection in Multi-timescale Data using Gaussian Processes**. KDD 2017: Anomaly Detection in Finance . [paper](http://proceedings.mlr.press/v71/adelsberg18a/adelsberg18a.pdf)
    
    *Matthew van Adelsberg, Christian Schwantes*
    
14. **Ensemble-based Anomaly Detection Using Cooperative Agreement**. KDD 2017: Anomaly Detection in Finance . [paper](http://proceedings.mlr.press/v71/kashef18a/kashef18a.pdf)
    
    *Rasha Kashef*
    
15. **Real-time anomaly detection system for time series at scale**. KDD 2017: Anomaly Detection in Finance . [paper](http://proceedings.mlr.press/v71/toledano18a/toledano18a.pdf)
    
    *Ira Cohen, Meir Toledano, Yonatan Ben Simhon, Inbal Tadeski*
    
16. **PD-FDS: Purchase Density based Online Credit Card Fraud Detection System**. KDD 2017: Anomaly Detection in Finance . [paper](http://proceedings.mlr.press/v71/ki18a/ki18a.pdf)
    
    *Youngjoon Ki, Ji Won Yoon*
    
17. **Deep Learning to Detect Treatment Fraud amongst Healthcare Providers**. KDD 2017: Anomaly Detection in Finance . [paper](http://proceedings.mlr.press/v71/lasaga18a/lasaga18a.pdf)
    
    *Daniel Lasaga, Prakash Santhana*

</details>
    

<!-- KDD 2019
[09:00 - 09:05 AM] Detection of Accounting Anomalies in the Latent Space using Adversarial Autoencoder Neural Networks - Marco Schreyer (University of St. Gallen); Timur Sattarov (Deutsche Bundesbank); Christian Schulze (University of St. Gallen); Bernd Reimer (PricewaterhouseCoopers GmbH WPG); Damian Borth (University of St. Gallen)
[09:05 - 09:10 AM] Risk Management via Anomaly Circumvent: Mnemonic Deep Learning for Midterm Stock Prediction - Xinyi Li (Columbia University); Yinchuan Li (Columbia University); Xiao-Yang Liu (Columbia University); Christina Dan Wang (New York University)
[09:10 - 09:15 AM] Case-Based Reasoning for Assisting Domain Experts in Processing Fraud Alerts of Black-Box Machine Learning Models - Hilde J.P. Weerts (Eindhoven University of Technology); Mykola Pechenizkiy (TU Eindhoven); Werner van Ipenburg (Cooperatieve Rabobank U.A.)
[09:15 - 09:20 AM] A framework for anomaly detection using language modeling, and its applications to finance - Armineh Nourbakhsh (S&P Global Ratings); Grace Bang (S&P Global)
[09:20 - 09:25 AM] Detecting Unusual Expense Groups for Customer Advice Apps - Axel Brando (BBVA Data & Analytics); Jose Rodriguez-Serrano (BBVA Analytics SL); Jordi Vitria (Universitat de Barcelona)
[09:25 - 09:30 AM] Empirical Study on Detecting Controversy in Social Media Zhiqiang Ma (S&P Global); Xiaomo Liu (S&P Global); Azadeh Nematzadeh (S&P Global); Grace Bang (S&P Global)
Automating Data Monitoring: Detecting Structural Breaks in Time Series Data Using Bayesian Minimum Description Length - Yingbo Li (Capital One)
Infusing domain knowledge in AI-based "black box" models for better explainability with application in bankruptcy prediction - Sheikh Rabiul Islam (Tennessee Technological University); William Eberle (Tennessee Tech. University); Sid Bundy (Tennessee Technological University); Sheikh Khaled Ghafoor (Tennessee Technological University)
Automatic Model Monitoring for Data Streams - Fábio Pinto (Feedzai); Marco O P Sampaio (Feedzai); Pedro Bizarro (Feedzai)
Online NEAT for Credit Evaluation - a Dynamic Problem with Sequential Data - Yue Liu (The Southern University of Science and Technology ); Adam Ghandar (Southern University of Science and Technology); Georgios Theodoropoulos (Southern University of Science and Technology)
Detecting Anomalies in Sequential Data with Higher-Order Networks - Mandana Saebi (University of Notre Dame)
Calibration for Anomaly Detection - Adrian Benton (Bloomberg L.P.)
Textual Outlier Detection and Anomalies in Financial Reporting - Leslie Barrett (Bloomberg L.P.)
Detecting anomalous doctors by measuring behavioral volatility using temporal clustering - Daniel A Lasaga (Deloitte)
Multi-Modal and Multi-Level Machine Learning for Fake Rideshare Trip Detection - Chengliang Yang (Uber Techologies)
Systematic detection of fraudulent account registration - Yun Zhang (Uber technologies, Inc)
Dynamic Linear Regression for Variable Level Monitoring - Thomas J Caputo (Capital One) -->

<!-- KDD 2020
Machine learning methods to detect money laundering in the Bitcoin blockchain in the presence of label scarcity, Joana Lorenz (NOVA-IMS); Maria Ines P P Silva (Feedzai)*; David Aparicio (Feedzai); Joao Ascesao (Feedzai); Pedro Bizarro (Feedzai) Paper #13 [Video]

On the Robustness of Deep Reinforcement Learning Based Optimal Trade Execution Systems, Siyu Lin (University of Virginia)*; Peter Beling (University of Virginia), Paper #15 [Video]

Detection of Balance Anomalies with Quantile Regression: the Power of Non-symmetry, David Muelas Recuenco (BBVA Data & Analytics)*; Luis Peinado (BBVA Data & Analytics); Axel Brando (BBVA Data & Analytics); Jose Rodriguez-Serrano (BBVA Analytics SL), Paper #20 [Video] [Paper]

Navigating the Dynamics of Financial Embeddings over Time, Alan O Salimov (Capital One)*; Brian Nguyen (Capital One); Antonia Gogoglou (Capital One); C Bayan Bruss (Capital One); Jonathan Rider (Capital One), Paper #18 [Video] [Paper]

Evolution of Credit Risk Using a Personalized Pagerank Algorithm for Multilayer Networks, Cristian Bravo (University of Western Ontario)*; María Óskarsdóttir (University of Reykjavík), Paper #11 [Video] [Paper]

Mitigating Bias in Online Microfinance Platforms: A Case Study on Kiva.org, Soumajyoti Sarkar (Arizona State University)*; Hamidreza Alvari (Arizona State University), Paper #29 [Video]

Accurate and Intuitive Contextual Explanations using Linear Model Trees, Aditya Lahiri (American Express)*; Narayanan U Edakunni (American Express AI Labs), Paper #3 [Video] [Paper]

Multi-stream RNN for Merchant Transaction Prediction, Zhongfang Zhuang (Visa Research)*; Michael Yeh (Visa Research); Liang Wang (Visa Research); Wei Zhang (Visa Research); Junpeng Wang (Visa Research), Paper #5 [Slides] [Video]

Financial Sentiment Analysis with Pre-trained Language Models, Dogu Araci (Prosus)*; Zulkuf Genc (Prosus), Paper # 9 [Video]

Adverse Media Mining for KYC and ESG Compliance, Rupinder P Khandpur (Virginia Tech); Ashit Talukder (Moody's Analytics); Rupinder Khandpur (Moodys Analytics)*, Paper #34 [Video]

CoronaPulse: Real-Time Sentiment Analysis and Emergent Multi-Sector Financial Risk Detection From CoVID-19 Events, Ashit Talukder (Moody's Analytics)*; Daulet Nurmanbetov (Moody's Analytics), Paper #28 [Video] [Paper]

A Scalable Framework for Group Fraud Detection in Transaction using Deep Graph Neural Network, Wei Min (eBay)*; Zhichao Han (eBay); Zitao Zhang (eBay); Shengqian Chen (eBay); Wenyu Dong (eBay); Yang Zhao (eBay), Paper #19 [Video] [Slides] [Paper]

A Unified Machine Learning Framework for Targeting Financial Product Offerings, Shankar Sankararaman (Intuit)*; Debasish Das (Intuit); Deepesh Ramachandran Vijayalekshmi (Intuit); Babak Aghazadeh (Intuit), Paper #16 [Video] [Paper]

Machine Learning for Temporal Data in Finance: Challenges and Opportunities, Jason Wittenbach (Capital One)*; C Bayan Bruss (Capital One); Brian d'Alessandro (Capital One), Paper #30 [Video]

Predicting Account Receivables with Machine Learning, Ana Paula Appel (IBM)*; Gabriel Malfatti (IBM Research); Renato Cunha (IBM); Bruno Lima (IBM Research); Rogerio de Paula (IBM Research, Brazil), Paper #6 [Video] [Paper]

Explainable Clustering and Application to Wealth Management Compliance, Enguerrand Horel (Stanford University)*; Kay Giesecke (Stanford University); Victor Storchan (J.P. Morgan); Naren Chittar (J.P. Morgan), Paper #7 [Video] [Paper]

FairXGBoost: Fairness-aware Classification in XGBoost, Srinivasan Ravichandran (American Express)*; Drona Khurana (American Express); Bharath Venkatesh (American Express); Narayanan U Edakunni (American Express AI Labs), Paper #23 [Video] [Paper]

Hierarchical Contextual Document Embeddings for Long Financial Text Regression, Vipula Rawte (Rensselaer Polytechnic Institute)*; Aparna Gupta (RPI); Mohammed Zaki (RPI), Paper #31 [Video] [Paper]

On the Optimal Baseline Auto Insurance Premium, Patrick Hosein (University of the West Indies)*, Paper #21 [Video] [Paper]

Personalized Welcome Messages in Conversational Chatbots via Time Aware Self-Attentive Models, Homa Foroughi (Intuit)*; Chang Liu (Intuit); Pankaj Gupta (Intuit), Paper #10 [Video] [Paper]

Data- and Model-driven Multi-Touch Attribution for Omnichannel Conversions, Yue Duan (Capital One)*; Jie Shen (Capital One), Paper #22 [Video] [Paper]

Dealing with missing Industry Codes: Imputation and Representation using Sequence Classification, Behrouz Saghafikhadem (Capital One); Jihan Wei (Capital One); Jiankun Liu (Capital One)*;Nickolas Wilson (Capital One); Ankur Mohan (Capital One), Paper #35 [Video] [Paper]

Towards Earnings Call and Stock Price Movement, Zhiqiang Ma (S&P Global)*; Grace Bang (S&P Global); Xiaomo Liu (S&P Global); Chong Wang (S&P Global ), Paper #2 [Video] [Paper]

Evidence of Rising Market Efficiency from Intra-day Equity Price Prediction, David Byrd (Ga Tech)*; Tucker Balch (JP Morgan), Paper #27 [Video] [Paper]

Alpha Discovery Neural Network, the Special Fountain of Financial Trading Signals, Jie Fang (Tsinghua University)*; Shutao Xia (Tsinghua University); Jianwu Lin (Tsinghua Shenzhen Graduate School); Zhikang Xia (Tsinghua University); Xiang Liu (Tsinghua Shenzhen International Graduate School); Yong Jiang (Tsinghua University), Paper #1 [Video] [Poster] [Paper] -->


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


<!-- ### [AAAI 2020: Knowledge Discovery from Unstructured Data in Financial Services]
Explainable Deep Behavioral Sequence Clustering for Transaction Fraud Detection
Wei Min, Weiming Liang, Hang Yin, Zhurong Wang, Alok Lal and Mei Li
Corporate Cyber-security Event Detection Platform
Zheng Nie, Jingjing Feng, Steve Pomerville and Azadeh Nematzadeh
Data Augmentation Methods for Reject Inference in Credit Risk Models
Jingxian Liao, Wei Wang, Jason Xue and Anthony Lei
AR-Stock: Deep Augmented Relational Stock Prediction
Tianxin Wei, Yuning You and Tianlong Chen
Building a Credit Risk Model using Transfer Learning and Domain Adaptation
Hendra Suryanto, Charles Guan, Ada Guan, Pengqian Li, Paul Compton, Michael Bain and Ghassan Beydoun
Boosting over Deep Learning for Earnings
Stephen Choi, Xinyue Cui and Jingran Zhao
Leaking Sensitive Financial Accounting Data in Plain Sight using Deep Autoencoder Neural Networks
Marco Schreyer, Christian Schulze and Damian Borth
Sensitive Data Detection with High-Throughput Neural Network Models for Financial Institutions
Anh Truong, Austin Walters and Jeremy Goodsitt
Knowledge discovery with Deep RL for selecting financial hedges
Eric Benhamou, David Saltiel, Sandrine Ungari, Abhishek Mukhopadhyay, Jamal Atif and Rida Laraki
A Comparison of Multi-View Learning Strategies for Satellite Image-Based Real Estate Appraisal
Jan-Peter Kucklick and Oliver Müller
PayVAE: A Generative Model for Financial Transactions
Niccolo Dalmasso, Robert Tillman, Prashant Reddy and Manuela Veloso -->


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
    <table style="margin-left: 80px; margin-right: auto;">
        <tr>
            <td>
                <img src="./figs/Zhihu.png" style="max-width: 50%;"/>
            </td>
            <td>
                <img src="./figs/OA.png" style="max-width: 50%;"/>
            </td>
        </tr>
    </table>
</html>
