# Частина 1.
```
# scripts/01_prepare_data.py
Читаємо датасет: 10000it [00:00, 65972.86it/s]
Завантажено статей:10000
Розподіл за категоріями (топ-10):
category
astro-ph              1838
hep-th                 680
hep-ph                 671
quant-ph               564
gr-qc                  350
cond-mat.mes-hall      307
cond-mat.str-el        292
cond-mat.mtrl-sci      291
cond-mat.stat-mech     271
math.AG                209
Name: count, dtype: int64
Розподіл за роками:
year
2007    10000
Name: count, dtype: int64
Приклад запису:
{'id': '0704.0001', 'title': 'Calculation of prompt diphoton production cross sections at Tevatron and   LHC energies', 'abstract': 'A fully differential calculation in perturbative quantum chromodynamics is presented for the production of massive photon pairs at hadron colliders. All next-to-leading order perturbative contributions from quark-antiquark, gluon-(anti)quark, and gluon-gluon subprocesses are included, as well as all-orders resummation of initial-state gluon radiation valid at next-to-next-to-leading logarithmic accuracy. The region of phase space is specified in which the calculation is most reliable. Good agreement is demonstrated with data from the Fermilab Tevatron, and predictions are made for more detailed tests with CDF and DO data. Predictions are shown for distributions of diphoton pairs produced at the energy of the Large Hadron Collider (LHC). Distributions of the diphoton pairs from the decay of a Higgs boson are contrasted with those produced from QCD processes at the LHC, showing that enhanced sensitivity to the signal can be obtained with judicious selection of events.', 'authors': 'BalázsC., BergerE. L., NadolskyP. M., YuanC. -P.', 'year': 2007, 'category': 'hep-ph'}
Збережено в../data/arxiv_subset.parquet
```

Відповіді на питання:

1. Чим Pinecone відрізняється від Qdrant і Chroma за моделлю розгортання, ліцензією і продуктивністю? У якому сценарії ви б обрали кожен із них?

Pinecone — це готовий хмарний сервіс для роботи з векторними базами даних. Його головна перевага полягає в тому, що не потрібно самостійно налаштовувати сервери та інфраструктуру. Qdrant є open-source рішенням, яке можна розгорнути на власному сервері або в хмарі та отримати більше контролю над системою. Chroma також є open-source базою даних, але вона орієнтована насамперед на навчальні проєкти, прототипування та невеликі застосунки. Для швидкого запуску продакшн-системи я б обрав Pinecone, для масштабованого власного розгортання — Qdrant, а для навчання або невеликого проєкту — Chroma.

2. Чому для задачі пошуку по науковим текстам обрана модель specter2_base, а не універсальна all-MiniLM-L6-v2? Знайдіть картку моделі на HuggingFace і процитуйте, для яких задач вона навчена.

Модель all-MiniLM-L6-v2 є універсальною. Вона чудово розрізняє побутові теми, тексти з Вікіпедії чи соцмереж, але пасує перед специфічним науковим лексиконом, формулами та специфічними зв'язками між статтями.

Наукові тексти мають унікальний контекст (цитування, академічний стиль). Модель specter2_base від AllenAI спеціально розроблялася під це.

Цитата з картки моделі на HuggingFace:
"SPECTER2 is a family of models that succeeds SPECTER and is capable of generating task specific embeddings for scientific tasks when paired with adapters. Given the combination of title and abstract of a scientific paper or a short textual query, the model can be used to generate effective embeddings to be used in downstream applications. SPECTER2 has been trained on over 6M triplets of scientific paper citations..."

Вона навчалася на триплетах цитувань (стаття-запит, стаття, яку вона цитує [+], і випадкова стаття [-]), тому вміє групувати наукові праці не лише за схожістю слів, а й за логікою наукового дискурсу.

3. Що написано у картці моделі про рекомендовану метрику схожості? Чому це важливо при створенні індексу?

У документації моделей сімейства SPECTER вказано, що базовим методом оцінки близькості векторів є косинусна схожість (Cosine Similarity) або скалярний добуток (Dot Product).  

Це критично важливо враховувати під час створення індексу у векторній базі (параметр metric="cosine"), адже якщо геометрія простору, в якому модель звикла виражати схожість, не збігатиметься з метрикою бази даних, результати пошуку перетворяться на випадковий шум.

```
# scripts/02_embed.py
No sentence-transformers model found with name allenai/specter2_base. Creating a new one with mean pooling.
Batches: 100%|███████████████████████████████████████████████████████████████████████████████████| 157/157 [26:08<00:00,  9.99s/it]
Загальна кількість оброблених текстів: 10000
Розмірність ембеддингів: 768
Норма першого ембеддингу: 1.000000
Ембеддинги успішно збережено у файл: ../embeddings/embeddings.npy
```

Відповідь на питання:
Поясніть, чому при використанні нормалізованих ембеддингів (одиничної довжини) косинусна схожість (cosine similarity) еквівалентна скалярному добутку (dot product)?

Після нормалізації всі вектори мають однакову довжину. У такому випадку косинусна схожість залежить лише від напрямку векторів і фактично дає той самий результат, що й скалярний добуток. Тому ранжування документів для цих двох метрик буде однаковим.


# Частина 2.
```
# scripts/03_load_to_pinecone.py
Індекс 'arxiv-papers' не знайдено. Створення нового серверлес індексу...
Індекс 'arxiv-papers' успішно створено.

Початок завантаження 10000 векторів в Pinecone батчами по 200...
Завантаження в Pinecone: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [04:39<00:00,  5.60s/it]
Завантаження успішно завершено!
Загальна кількість векторів в індексі 'arxiv-papers': 10000
```

# Частина 3.
```
# scripts/04_search.py
No sentence-transformers model found with name allenai/specter2_base. Creating a new one with mean pooling.

Виконуємо чистий семантичний пошук для запиту: 'teaching machines to recognize objects in pictures'

Результати чистого семантичного пошуку (Pinecone):
1. [cond-mat.soft] (2007) Capturing knots in polymers
   Abstract snippet: This paper visualizes a knot reduction algorithm...
   Score/Distance: 0.8288
2. [physics.ins-det] (2007) Symbolic sensors : one solution to the numerical-symbolic interface
   Abstract snippet: This paper introduces the concept of symbolic sensor as an extension of the smart sensor one. Then, ...
   Score/Distance: 0.8263
3. [math.HO] (2007) The Mathematics
   Abstract snippet: This is an essay that considering the knowledge structure and language of a different nature, attemp...
   Score/Distance: 0.8256
4. [physics.comp-ph] (2007) Modeling the field of laser welding melt pool by RBFNN
   Abstract snippet: Efficient control of a laser welding process requires the reliable prediction of process behavior. A...
   Score/Distance: 0.8170
5. [quant-ph] (2007) Why should anyone care about computing with anyons?
   Abstract snippet: In this article we present a pedagogical introduction of the main ideas and recent advances in the a...
   Score/Distance: 0.8146
   
Фільтр А: Відсутні статті по reinforcement learning за останні 5 років і категорія cs.LG

Фільтр B: статті по reinforcement learning до 2015 року:
1. [cs.MA] (2007) Multi-Agent Modeling Using Intelligent Agents in the Game of Lerpa
   Abstract snippet: Game theory has many limitations implicit in its application. By utilizing multiagent modeling, it i...
   Score/Distance: 0.8445
2. [cond-mat.stat-mech] (2007) Introduction to Phase Transitions in Random Optimization Problems
   Abstract snippet: Notes of the lectures delivered in Les Houches during the Summer School on Complex Systems (July 200...
   Score/Distance: 0.8194
3. [cs.NE] (2007) Architecture for Pseudo Acausal Evolvable Embedded Systems
   Abstract snippet: Advances in semiconductor technology are contributing to the increasing complexity in the design of ...
   Score/Distance: 0.8102
4. [physics.pop-ph] (2007) Why only few are so successful ?
   Abstract snippet: In many professons employees are rewarded according to their relative performance. Corresponding eco...
   Score/Distance: 0.8010
5. [physics.soc-ph] (2007) Opinion Dynamics and Sociophysics
   Abstract snippet: No abstract given. Contents:   I. Definition and Introduction   II. Schelling Model   III. Opinion D...
   Score/Distance: 0.7993

Пошук: COSINE SIMILARITY:
1. [cond-mat.soft] (2007) Capturing knots in polymers
   Abstract snippet: This paper visualizes a knot reduction algorithm...
   Score/Distance: 0.8294
2. [physics.ins-det] (2007) Symbolic sensors : one solution to the numerical-symbolic interface
   Abstract snippet: This paper introduces the concept of symbolic sensor as an extension of the smart sensor one. Then, ...
   Score/Distance: 0.8260
3. [math.HO] (2007) The Mathematics
   Abstract snippet: This is an essay that considering the knowledge structure and language of a different nature, attemp...
   Score/Distance: 0.8254
4. [physics.comp-ph] (2007) Modeling the field of laser welding melt pool by RBFNN
   Abstract snippet: Efficient control of a laser welding process requires the reliable prediction of process behavior. A...
   Score/Distance: 0.8181
5. [nlin.CD] (2007) Python for Education: Computational Methods for Nonlinear Systems
   Abstract snippet: We describe a novel, interdisciplinary, computational methods course that uses Python and associated...
   Score/Distance: 0.8142

Пошук: DOT PRODUCT:
1. [cond-mat.soft] (2007) Capturing knots in polymers
   Abstract snippet: This paper visualizes a knot reduction algorithm...
   Score/Distance: 0.8294
2. [physics.ins-det] (2007) Symbolic sensors : one solution to the numerical-symbolic interface
   Abstract snippet: This paper introduces the concept of symbolic sensor as an extension of the smart sensor one. Then, ...
   Score/Distance: 0.8260
3. [math.HO] (2007) The Mathematics
   Abstract snippet: This is an essay that considering the knowledge structure and language of a different nature, attemp...
   Score/Distance: 0.8254
4. [physics.comp-ph] (2007) Modeling the field of laser welding melt pool by RBFNN
   Abstract snippet: Efficient control of a laser welding process requires the reliable prediction of process behavior. A...
   Score/Distance: 0.8181
5. [nlin.CD] (2007) Python for Education: Computational Methods for Nonlinear Systems
   Abstract snippet: We describe a novel, interdisciplinary, computational methods course that uses Python and associated...
   Score/Distance: 0.8142

Пошук: L2 DISTANCE:
1. [cond-mat.soft] (2007) Capturing knots in polymers
   Abstract snippet: This paper visualizes a knot reduction algorithm...
   Score/Distance: 0.5842
2. [physics.ins-det] (2007) Symbolic sensors : one solution to the numerical-symbolic interface
   Abstract snippet: This paper introduces the concept of symbolic sensor as an extension of the smart sensor one. Then, ...
   Score/Distance: 0.5899
3. [math.HO] (2007) The Mathematics
   Abstract snippet: This is an essay that considering the knowledge structure and language of a different nature, attemp...
   Score/Distance: 0.5910
4. [physics.comp-ph] (2007) Modeling the field of laser welding melt pool by RBFNN
   Abstract snippet: Efficient control of a laser welding process requires the reliable prediction of process behavior. A...
   Score/Distance: 0.6032
5. [nlin.CD] (2007) Python for Education: Computational Methods for Nonlinear Systems
   Abstract snippet: We describe a novel, interdisciplinary, computational methods course that uses Python and associated...
   Score/Distance: 0.6095
``` 

Відповіді на питання:

1. Чи збігаються топ-5 для cosine і dot product і чому?

Так, тому що використовуються нормалізовані ембеддинги. У цьому випадку обидві метрики оцінюють схожість однаково, тому результати пошуку та порядок документів збігаються.

2. Чи відрізняються результати для L2 і чому?

Ні, в нашому не відрізніняються. Для нормалізованих векторів результати зазвичай дуже схожі, але не завжди ідентичні. L2 оцінює відстань між векторами, а не лише напрямок, тому в окремих випадках порядок документів може трохи відрізнятися.

3. Що сталося б, якби ембеддинги не були нормалізовані?

Тоді на результат пошуку впливала б не лише семантична схожість, а й довжина векторів. Документи з більшими значеннями векторів могли б отримувати вищі позиції навіть тоді, коли вони менш релевантні за змістом.

4. Пошук з фільтрацією: порівняти видачу і пояснити відмінності.

В результат прикладу А не попадають ніякі статті, так першим скриптом (01_prepare_data.py) тільки 10000 перших статей відібрано, приклад Б - перших 5 статей, що відповідають умовам фільтру. 


# Частина 4
```
# scripts/05_chunking.py
Вибрано 30 статей з найдовшими анотаціями.
Створення індексу 'arxiv-chunks-fixed'...
Створення індексу 'arxiv-chunks-semantic'...

Обробка за допомогою стратегії: FIXED
Завантаження 254 чанків в індекс...
Upsert fixed: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:10<00:00,  3.36s/it]

Обробка за допомогою стратегії: SEMANTIC
Завантаження 193 чанків в індекс...
Upsert semantic: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:04<00:00,  2.08s/it]

Пошук за запитом: 'computational methods for nonlinear systems'

Результати для індексу: arxiv-chunks-fixed
1. [Score: 0.8085] Spin Effects in Quantum Chromodynamics and Recurrence Lattices with   Multi-Site Exchanges
   [Chunk #1.0] Text: our approach is the method of recursive (hierarchical) lattices. We apply the method of dynamical ma...
2. [Score: 0.8043] Spin Effects in Quantum Chromodynamics and Recurrence Lattices with   Multi-Site Exchanges
   [Chunk #0.0] Text: In this thesis, we consider some spin effects in QCD and recurrence lattices with multi-site exchang...
3. [Score: 0.8042] Spin Evolution of Accreting Neutron Stars: Nonlinear Development of the   R-mode Instability
   [Chunk #0.0] Text: The nonlinear saturation of the r-mode instability and its effects on the spin evolution of Low Mass...
4. [Score: 0.7952] The Boundary Conditions of the Heliosphere: Photoionization Models   Constrained by Interstellar and In Situ Data
   [Chunk #2.0] Text: Bubble (LB) plasma, assuming that the cloud is evaporating because of thermal conduction. We create ...
5. [Score: 0.7935] A Systematic Study of the Final Masses of Gas Giant Planets
   [Chunk #8.0] Text: We derive analytic formulae for the final masses in the different regions and the locations of the b...

Результати для індексу: arxiv-chunks-semantic
1. [Score: 0.8131] Rotation and activity of pre-main-sequence stars
   [Chunk #6.0] Text: (abridged)...
2. [Score: 0.8023] A Systematic Study of the Final Masses of Gas Giant Planets
   [Chunk #1.0] Text: We first derive an analytical formula for surface density profile near the planetary orbit from cons...
3. [Score: 0.8017] Spin Evolution of Accreting Neutron Stars: Nonlinear Development of the   R-mode Instability
   [Chunk #0.0] Text: The nonlinear saturation of the r-mode instability and its effects on the spin evolution of Low Mass...
4. [Score: 0.7972] Ages for illustrative field stars using gyrochronology: viability,   limitations and errors
   [Chunk #1.0] Text: The technique is clarified and developed here, and used to derive ages for illustrative groups of ne...
5. [Score: 0.7969] Spin Effects in Quantum Chromodynamics and Recurrence Lattices with   Multi-Site Exchanges
   [Chunk #0.0] Text: In this thesis, we consider some spin effects in QCD and recurrence lattices with multi-site exchang...

Пошук за запитом: 'mathematical analysis and algebraic structures'

Результати для індексу: arxiv-chunks-fixed
1. [Score: 0.8328] Spin Effects in Quantum Chromodynamics and Recurrence Lattices with   Multi-Site Exchanges
   [Chunk #0.0] Text: In this thesis, we consider some spin effects in QCD and recurrence lattices with multi-site exchang...
2. [Score: 0.8281] Spin Effects in Quantum Chromodynamics and Recurrence Lattices with   Multi-Site Exchanges
   [Chunk #1.0] Text: our approach is the method of recursive (hierarchical) lattices. We apply the method of dynamical ma...
3. [Score: 0.8067] Spin Effects in Quantum Chromodynamics and Recurrence Lattices with   Multi-Site Exchanges
   [Chunk #3.0] Text: consider the recurrent models of $^3$He defined on the square, Husimi and hexagon lattices. Using th...
4. [Score: 0.8046] Spin Effects in Quantum Chromodynamics and Recurrence Lattices with   Multi-Site Exchanges
   [Chunk #2.0] Text: macromolecules. First, we analyze the helix-coil phase transition for polypeptides and proteins, and...
5. [Score: 0.7978] Conjectures on exact solution of three - dimensional (3D) simple   orthorhombic Ising lattices
   [Chunk #8.0] Text: phenomenon differs with the 2D to 1D crossover phenomenon and there is a gradual crossover of the ex...

Результати для індексу: arxiv-chunks-semantic
1. [Score: 0.8426] Rotation and activity of pre-main-sequence stars
   [Chunk #6.0] Text: (abridged)...
2. [Score: 0.8305] Spin Effects in Quantum Chromodynamics and Recurrence Lattices with   Multi-Site Exchanges
   [Chunk #0.0] Text: In this thesis, we consider some spin effects in QCD and recurrence lattices with multi-site exchang...
3. [Score: 0.8055] Spin Effects in Quantum Chromodynamics and Recurrence Lattices with   Multi-Site Exchanges
   [Chunk #1.0] Text: We apply the method of dynamical mapping (or recursive lattices) for investigation of magnetic prope...
4. [Score: 0.8054] Ages for illustrative field stars using gyrochronology: viability,   limitations and errors
   [Chunk #1.0] Text: The technique is clarified and developed here, and used to derive ages for illustrative groups of ne...
5. [Score: 0.8049] Spin Effects in Quantum Chromodynamics and Recurrence Lattices with   Multi-Site Exchanges
   [Chunk #2.0] Text: Next we consider the recurrent models of $^3$He defined on the square, Husimi and hexagon lattices. ...
```

Відповіді на питання:

1. Яка стратегія дає більш осмислені чанки?

Semantic Chunking. Цей підхід розбиває текст на межах речень. Завдяки цьому кожна смислова одиниця (чанк) містить закінчену думку, де зрозуміло, хто є суб'єктом дії, який контекст і який висновок робиться.

2. Чи є випадки розрізаних речень і як це впливає на ембеддинги?

У стратегії Fixed-size chunking речення розрізаються постійно, оскільки алгоритм сліпо рахує слова (наприклад, рівно 50 слів).
Вплив на ембеддинги: Негативний. Якщо відірвати половину речення, механізм уваги (attention) трансформера не зможе правильно зчитати логічний зв'язок. Вектор такого розірваного шматка стає «семантично розмитим», втрачає точність і погіршує якість як пошуку, так і подальшої генерації (RAG).

3. Як розмір overlap впливає на кількість чанків і покриття тексту?

Кількість чанків: Що більший overlap, то більше чанків утворюється (крок повзунка зменшується, текст сильніше дублюється).
Покриття тексту: Великий overlap мінімізує ризик втрати контексту на "стиках" чанків. Якщо важлива думка записана якраз на межі розрізу, завдяки перекриттю вона повністю потрапить або в попередній, або в наступний чанк у неспотвореному вигляді.


# Частина 5
```
# scripts/06_hybrid_search.py 
Запит №1: 'BERT fine-tuning'

   Метод: Точний пошук BM25
      1. (2007) [hep-ph] The NMSSM Solution to the Fine-Tuning Problem, Precision Electroweak   Constr...
      2. (2007) [hep-th] Fine-Tuning in Brane-antibrane Inflation
      3. (2007) [hep-ph] Conformal dynamics in gauge theories via non-perturbative   renormalization g...
      4. (2007) [hep-lat] Inverse Monte-Carlo determination of effective lattice models for SU(3)   Yan...
      5. (2007) [hep-th] Eternal Inflation is "Expensive"

   Метод: Векторний пошук (Pinecone)
      1. (2007) [math.CO] Misere quotients for impartial games: Supplementary material
      2. (2007) [cond-mat.stat-mech] Introduction to Phase Transitions in Random Optimization Problems
      3. (2007) [math.FA] Abstract Convexity and Cone-Vexing Abstractions
      4. (2007) [math.CO] The Compositions of the Differential Operations and Gateaux Directional   Der...
      5. (2007) [quant-ph] Experimental local realism tests without fair sampling assumption

   Метод: Гібридний пошук
      1. [RRF=0.0164] (2007) [math.CO] Misere quotients for impartial games: Supplementary material
      2. [RRF=0.0164] (2007) [hep-ph] The NMSSM Solution to the Fine-Tuning Problem, Precision Electroweak   Constr...
      3. [RRF=0.0161] (2007) [cond-mat.stat-mech] Introduction to Phase Transitions in Random Optimization Problems
      4. [RRF=0.0161] (2007) [hep-th] Fine-Tuning in Brane-antibrane Inflation
      5. [RRF=0.0159] (2007) [math.FA] Abstract Convexity and Cone-Vexing Abstractions

Запит №2: 'Yann LeCun convolutional networks'

   Метод: Точний пошук BM25
      1. (2007) [cs.IT] On Punctured Pragmatic Space-Time Codes in Block Fading Channel
      2. (2007) [cs.IT] Trellis-Coded Quantization Based on Maximum-Hamming-Distance Binary   Codes
      3. (2007) [cond-mat.dis-nn] Response of degree-correlated scale-free networks to stimuli
      4. (2007) [cond-mat.dis-nn] Numerical evaluation of the upper critical dimension of percolation in   scal...
      5. (2007) [physics.soc-ph] On Automorphism Groups of Networks

   Метод: Векторний пошук (Pinecone)
      1. (2007) [math.ST] Multilayer Perceptron with Functional Inputs: an Inverse Regression   Approach
      2. (2007) [cs.NI] The Netsukuku network topology
      3. (2007) [math.CO] The Compositions of the Differential Operations and Gateaux Directional   Der...
      4. (2007) [physics.comp-ph] Modeling the field of laser welding melt pool by RBFNN
      5. (2007) [math.OC] Adaptive classification of temporal signals in fixed-weights recurrent   neur...

   Метод: Гібридний пошук
      1. [RRF=0.0303] (2007) [cond-mat.stat-mech] Optimization in Gradient Networks
      2. [RRF=0.0164] (2007) [math.ST] Multilayer Perceptron with Functional Inputs: an Inverse Regression   Approach
      3. [RRF=0.0164] (2007) [cs.IT] On Punctured Pragmatic Space-Time Codes in Block Fading Channel
      4. [RRF=0.0161] (2007) [cs.NI] The Netsukuku network topology
      5. [RRF=0.0161] (2007) [cs.IT] Trellis-Coded Quantization Based on Maximum-Hamming-Distance Binary   Codes

Запит №3: 'making computers understand human emotions from text'

   Метод: Точний пошук BM25
      1. (2007) [cs.HC] An Automated Evaluation Metric for Chinese Text Entry
      2. (2007) [cs.CL] On the Development of Text Input Method - Lessons Learned
      3. (2007) [q-bio.GN] Towards Understanding the Origin of Genetic Languages
      4. (2007) [q-fin.TR] Detecting anchoring in financial markets
      5. (2007) [quant-ph] Database Manipulation on Quantum Computers

   Метод: Векторний пошук (Pinecone)
      1. (2007) [physics.soc-ph] Opinion Dynamics and Sociophysics
      2. (2007) [cs.CL] On the Development of Text Input Method - Lessons Learned
      3. (2007) [physics.soc-ph] Extracting the hierarchical organization of complex systems
      4. (2007) [cs.CY] Novelty and Collective Attention
      5. (2007) [cs.HC] Narratives within immersive technologies

   Метод: Гібридний пошук
      1. [RRF=0.0323] (2007) [cs.CL] On the Development of Text Input Method - Lessons Learned
      2. [RRF=0.0164] (2007) [physics.soc-ph] Opinion Dynamics and Sociophysics
      3. [RRF=0.0164] (2007) [cs.HC] An Automated Evaluation Metric for Chinese Text Entry
      4. [RRF=0.0159] (2007) [physics.soc-ph] Extracting the hierarchical organization of complex systems
      5. [RRF=0.0159] (2007) [q-bio.GN] Towards Understanding the Origin of Genetic Languages
```

Відповіді на питання:

1. Який метод дав кращий результат і чому?

Гібридний пошук.  Жоден з поодиноких методів (ні суто текстовий BM25, ні суто векторний Pinecone) не зміг показати ідеальний результат на всіх трьох запитахтак, тоді як гібридний враховує як точні збіги ключових слів, так і змістовну близькість документів, тому показує кращий результат.

2. Чи є документи в топ-5 гібридного пошуку, яких немає в топ-5 окремих методів, і чому?

Так, в нашому випадку це 2 запит. Документ може займати середні позиції в кожному окремому рейтингу, але після об'єднання результатів отримати достатньо високий сумарний рейтинг і потрапити до топ-5.

3. Як зміна параметра k в RRF впливає на видачу (наприклад, k=60 vs k=1)?

При k = 60: Вага позицій падає плавно. Алгоритм збалансовано оцінює всю першу десятку або сотню документів і активно заохочує консенсус між моделями.При k = 1: Штраф за втрату позиції стає жорстким (між 1 та 2 місцем різниця величезна). Алгоритм перетворюється на «радикала», який чує лише абсолютних лідерів (топ-1, топ-2) з кожного списку, повністю ігноруючи середню ланку, де моделі були згодні між собою.


# Частина 6

1. Семантичний пошук vs BM25. Наведіть конкретні приклади запитів із вашої роботи, де кожен метод виграв. Сформулюйте загальне правило: для яких типів запитів варто надати перевагу кожному з них?

Запит №1 "BERT fine-tuning": тут виграв BM25, він видав статті з "Fine-Tuning" (Наприклад, "The NMSSM Solution to the Fine-Tuning Problem, Precision Electroweak   Constr..."), це релевантний текстовий збіг, хоч і теми фізики. Для моделі SPECTER слово "BERT" у 2007 році було абсолютно невідомим (unknown токеном). Через це вектор запиту змістився у випадкову абстрактну зону, і база Pinecone видала комбінаторику й математику (math.CO), які взагалі не стосуються теми. Загалом для цього запиту релевантних результатів не буде, однак все-таки перемагає BM25.

Запит №2 "Yann LeCun convolutional networks": тут виграє семантичний пошук. Модель зрозуміла концепт машинного навчання за словом "networks" та контекстом. Вона видала Multilayer Perceptron (math.ST) та recurrent neural networks (math.OC). Тобто векторний пошук знайшов нейронні мережі, що є абсолютно правильним семантичним доменом для Янна ЛеКуна.

Сформулюємо правило: BM25 - коли запит містить конкретні власні назви, імена, бренди або абревіатури, є строгі ідентифікатори (номери моделей, коди помилок, артикули, назви законів чи специфічні терміни); семантичний пошук - запит сформульовано описово, «своїми словами», важливий синонімічний ряд, або коли користувач може робити друкарські помилки або використовувати інші граматичні форми слів, які текстовий пошук зазвичай пропускає.

2. Вплив розміру чанка. Що відбувається з якістю пошуку, якщо чанк занадто маленький (10–15 слів)? Якщо занадто великий (500+ слів)? Чи є оптимальний розмір або він залежить від задачі?

Занадто малий (10–15 слів): Вектор втрачає контекст. У чанк може потрапити підрядне речення без головного, модель не зрозуміє, до чого воно відноситься (втрата займенників та суб'єктів). Якість пошуку падає.
Занадто великий (500+ слів): Семантика розмивається, він може містити кілька різних тем, через що ембеддинг стане менш точним.
Оптимальний розмір: Залежить від завдання, але золотим стандартом для RAG вважається розмір у 100–250 слів (приблизно 1–2 абзаци), що дозволяє зберегти фокус і не втратити суть інформації.

3. Невідповідна метрика. Що сталося б, якби ми створили індекс Pinecone з метрикою euclidean (L2), але використовували модель, яка повертає нормалізовані вектори? Обґрунтуйте відповідь математично: виведіть зв’язок між L2 і cosine для одиничних векторів.

Для одиничних векторів:

∣∣A−B∣∣^2 = ∣∣A∣∣^2 + ∣∣B∣∣ ^ 2 − 2(A⋅B)

Оскільки:

||A|| = 1
||B|| = 1

отримуємо:

∣∣A−B∣∣^2 = 2 − 2cos(θ)

Тобто між L2 і cosine існує прямий математичний зв'язок.

Тому для нормалізованих векторів результати пошуку за cosine і L2 будуть дуже схожими, проте L2 оцінює відстань між векторами, а не лише напрямок, тому в окремих випадках порядок документів може трохи відрізнятися.

4. Обмеження Pinecone Starter. З якими обмеженнями безкоштовного тіру ви зіткнулися (або могли б зіткнутися)? Як би ви вирішили задачу, якби датасет був не 10000, а 10 мільйонів статей?

Обмеження безкоштовного тарифу (Starter plan):
- Жорсткий ліміт на обсяг сховища (максимум 1 GB на організацію) та обмеження за кількістю векторів (комфортно вміщує до 100 000 векторів розмірністю 768). Наш датасет на 10k статей пройшов легко, але чанкінг вже сильніше забиває пам'ять.
- Обмеження за пропускною здатністю: ліміти на кількість операцій читання/запису (Read/Write Units) на місяць, а також ліміт у 100 запитів на секунду (RPS) на рівні неймспейсу, через що скрипти масового завантаження без затримок можуть викликати помилку 429 Too Many Requests.

В мене проблем не виникло, однак якщо замість 10 тисяч статей потрібно було б працювати з 10 мільйонами, доцільно використати платний тариф або власне розгортання Qdrant, застосовувати ефективні індекси, квантизацію векторів та розподіл даних між кількома вузлами, щоб забезпечити прийнятну швидкість пошуку та зберігання.

