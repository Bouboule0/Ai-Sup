# Guide Complet d'Utilisation et de Benchmark — Ai-Sup 

> **Version anglaise disponible ici :** [docs/GUIDE_EN.md](./GUIDE_EN.md)

Bienvenue dans la documentation complète d'**Ai-Sup**. Ce guide détaille comment installer, utiliser, personnaliser et benchmarker l'orchestrateur **Plan-then-Execute (P-t-E)** multi-modèles.

---

## Sommaire

1. [Concept : Pourquoi Plan-then-Execute ?](#1-concept--pourquoi-plan-then-execute-)
2. [Installation et Configuration](#2-installation-et-configuration)
3. [Comment utiliser l'outil au quotidien (Commandes CLI)](#3-comment-utiliser-loutil-au-quotidien-commandes-cli)
4. [Comment lancer et personnaliser un Benchmark](#4-comment-lancer-et-personnaliser-un-benchmark)
5. [Personnaliser les modèles, couples et règles de routage](#5-personnaliser-les-modèles-couples-et-règles-de-routage)
6. [Choses importantes à savoir (Sécurité, État, Budget, Bonnes Pratiques)](#6-choses-importantes-à-savoir-sécurité-état-budget-bonnes-pratiques)
7. [Dépannage (FAQ)](#7-dépannage-faq)

---

## 1. Concept : Pourquoi Plan-then-Execute ?

Lorsqu'un agent IA développe un projet logiciel :
- **La réflexion architecturale (5 % à 10 % des tokens)** nécessite un modèle de frontière très puissant (ex. `Gemini 4 Argon` ou `Claude Opus 5.5`) capable de découper un problème complexe en étapes atomiques ordonnées avec des critères de succès stricts.
- **L'exécution répétitive (90 % à 95 % des tokens)** consiste à lire des fichiers, écrire du code ciblé, lancer `pytest` et analyser la sortie console dans une boucle **ReAct**.

Utiliser un modèle de frontière pour 100 % des étapes coûte **2 à 4 fois plus cher** et s'avère beaucoup plus lent. **Ai-Sup** délègue donc :
1. **Le Plan (`Plan`)** au modèle **Planner (Parent)**.
2. **L'Exécution (`Step` par `Step`)** au modèle **Executor (Enfant)** plus rapide et économique.
3. **L'Escalade intelligente (`StepRouter`)** : si l'Exécuteur échoue plusieurs fois de suite sur une étape ardue (par défaut `>= 2` échecs), le routeur redonne automatiquement la main au Planificateur pour replanifier l'étape, tant que le budget restant le permet.

---

## 2. Installation et Configuration

### Prérequis
- **Python 3.11+** (testé avec Python 3.12)
- Une clé API pour les appels réels. *(Aucune clé API n'est requise pour lancer le benchmark simulé ou les tests unitaires).*

### Installation standard (recommandée)

Depuis la racine du dépôt (`C:\Ai-Supervisor`) :

```powershell
# 1. (Optionnel) Créer et activer un environnement virtuel
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Sous Windows PowerShell
# source .venv/bin/activate    # Sous Linux / macOS

# 2. Installer le projet en mode éditable avec les outils de test
pip install -e ".[dev]"

# 3. Configurer vos clés API
Copy-Item .env.example .env
```

Ouvrez ensuite le fichier `.env` et renseignez vos clés :

```dotenv
GEMINI_API_KEY=AIzaSy...
ANTHROPIC_API_KEY=sk-ant-...
```

> **Astuce sans installation (`pip install`)** : Si les bibliothèques (`pydantic`, `typer`, `rich`, `pyyaml`, `httpx`, `google-genai`, `python-dotenv`) sont déjà présentes dans votre environnement Python/Conda, vous pouvez préfixer vos commandes par `$env:PYTHONPATH="src"; python -m orchestrator.cli ...` au lieu de `orch ...`.

---

## 3. Comment utiliser l'outil au quotidien (Commandes CLI)

Une fois installé, l'outil s'utilise via la commande `orch`.

### 3.1 Lister les couples d'IA disponibles (`orch pairings`)

Affiche tous les parentages `Planner -> Executor` configurés dans `config/models.yaml` ainsi que leurs tarifs par million de tokens (entrée / sortie) :

```powershell
orch pairings
```

### 3.2 Générer uniquement un plan d'exécution (`orch plan`)

Utile pour vérifier comment le Planificateur découpe votre objectif avant de modifier le moindre fichier :

```powershell
# Avec le planificateur par défaut (Gemini 4 Argon)
orch plan --goal "Créer une API REST minimaliste de gestion de tâches avec tests unitaires"

# En choisissant explicitement Claude Opus 5.5 comme planificateur
orch plan --goal "Refactoriser le module de parsing CSV" --planner claude-opus-5.5 --max-budget 1.00
```

### 3.3 Exécuter un projet complet (`orch run`)

Lance le cycle complet : **Planification initiale → Exécution étape par étape (boucle ReAct) → Escalade éventuelle → Rapport de coût final**.

```powershell
# Utiliser le couple par défaut (argon-flash) avec un plafond de 1.50 $
orch run --goal "Créer un convertisseur CSV vers JSON en CLI avec tests" --project ./examples/demo_project --max-budget 1.50

# Utiliser un couple préconfiguré spécifique (--pairing)
orch run --goal "Créer un solveur de graphe" --project ./examples/demo_project --pairing opus-sonnet --max-budget 2.50

# Combiner librement n'importe quel Planner et Executor (--planner et --executor)
orch run --goal "Ajouter la validation Pydantic" --project ./examples/demo_project --planner gemini-4-argon --executor gemini-3.1-pro --max-budget 2.00
```

### 3.4 Mode interactif (`--interactive`)

Si vous souhaitez valider manuellement chaque étape du plan avant que l'Exécuteur ne touche à vos fichiers :

```powershell
orch run --goal "Ajouter des tests unitaires" --project ./examples/demo_project --pairing argon-flash --interactive
```

À chaque étape, le terminal vous demandera : `Exécuter l'étape 'step_1' (...) ? [y/N]`. Si vous répondez `n`, l'étape est ignorée (`SKIPPED`).

---

## 4. Comment lancer et personnaliser un Benchmark

Le dépôt intègre un moteur de benchmark (`src/orchestrator/benchmark.py`) permettant de comparer objectivement les architectures **100 % Modèle Frontier** face aux architectures **Plan-then-Execute déléguées** sur un pipeline réaliste de 10 étapes.

### 4.1 Lancer le benchmark comparatif dans le terminal

Avec `orch` installé :
```powershell
orch benchmark
```

Ou directement avec Python sans installation préalable :
```powershell
$env:PYTHONPATH="src"
python -m orchestrator.cli benchmark
```

### 4.2 Exporter le tableau comparatif en Markdown

Pour sauvegarder le tableau complet dans un fichier `.md` :

```powershell
orch benchmark --output-md benchmark_results.md
```

### 4.3 Ajouter votre propre couple d'IA au benchmark

1. Ouvrez `config/models.yaml`.
2. Sous la clé `pairings:`, ajoutez une nouvelle entrée, par exemple `pro-flash` :
   ```yaml
   pairings:
     pro-flash:
       description: "Mid-Tier Gemini P-t-E (Gemini 3.1 Pro -> Gemini 3.8 Flash)"
       planner: "gemini-3.1-pro"
       executor: "gemini-3.8-flash"
       baseline: "argon-only"
   ```
3. Relancez `orch benchmark` : le couple `pro-flash` est automatiquement évalué et comparé à la référence `argon-only`.

### 4.4 Ajuster les profils de simulation (`SIMULATION_PROFILES`)

Dans `src/orchestrator/benchmark.py`, le dictionnaire `SIMULATION_PROFILES` définit pour chaque modèle :
- `plan_input_tokens` / `plan_output_tokens` : volume de tokens d'une planification.
- `exec_input_tokens_per_turn` / `exec_output_tokens_per_turn` : volume de tokens par tour ReAct.
- `exec_turns_per_step` : nombre moyen de tours ReAct par étape.
- `complex_step_failures_before_replan` : nombre d'échecs simulés sur l'étape algorithmique difficile (`step_4`) avant de déclencher une escalade vers le Planner via le `StepRouter`.

### 4.5 Comparer deux couples en conditions réelles (Live API)

Pour comparer deux couples sur vos propres clés API avec un vrai code généré :

```powershell
# 1. Créer deux dossiers de test propres
New-Item -ItemType Directory -Force ./examples/test_argon_flash, ./examples/test_argon_only

# 2. Lancer le couple délégué Argon -> Flash
orch run --goal "Write a fib.py module with memoized fibonacci and test_fib.py" --project ./examples/test_argon_flash --pairing argon-flash --max-budget 1.00

# 3. Lancer le baseline 100% Argon sur le même objectif
orch run --goal "Write a fib.py module with memoized fibonacci and test_fib.py" --project ./examples/test_argon_only --pairing argon-only --max-budget 2.00
```

Comparez ensuite le panneau récapitulatif final (**Coût Planner**, **Coût Executor**, **Coût total**) ainsi que les fichiers générés dans chaque dossier.

---

## 5. Personnaliser les modèles, couples et règles de routage

Toute la configuration est centralisée dans le dossier `config/` :

### 5.1 `config/models.yaml`
Contient :
- **`models:`** : chaque modèle avec son `id` API exact, son `provider` (`google` ou `anthropic`), et ses coûts `input_cost_per_million` et `output_cost_per_million` en dollars USD.
- **`pairings:`** : les couples préconfigurés utilisables avec `--pairing`.

### 5.2 `config/router.yaml`
Contient :
- **`max_retries_before_escalation: 2`** : au bout de 2 échecs de l'Exécuteur sur la même étape, le `StepRouter` force une escalade vers le Planificateur pour replanifier.
- **`min_remaining_budget_for_planner: 0.25`** : si le budget restant passe sous `0.25 $`, le routeur interdit toute escalade coûteuse vers le Planificateur et force l'Exécuteur à terminer avec le budget restant.
- **`shell.allowed_commands`** : liste blanche stricte des préfixes de commandes autorisés pour l'outil `run_command`.

---

## 6. Choses importantes à savoir (Sécurité, État, Budget, Bonnes Pratiques)

### 6.1 Persistance de l'état (`state.json`)
- Lors d'un `orch run --project ./mon_projet`, l'orchestrateur crée et met à jour en temps réel un fichier `mon_projet/state.json`.
- Ce fichier conserve :
  - `completed_steps` : les identifiants des étapes déjà validées.
  - `step_retries` : le compteur d'échecs par étape.
  - `step_outputs` : les résumés produits à chaque étape.
- **Bon à savoir** : Si vous relancez un nouvel objectif complètement différent dans le **même** dossier `--project`, supprimez d'abord `mon_projet/state.json` pour repartir d'un état vierge.

### 6.2 Sécurité et bac à sable (Sandboxing)
- **Confinement des fichiers (`tools/file_system.py`)** : Les outils `read_file`, `write_file` et `list_dir` interdisent formellement les chemins contenant `..`, les chemins absolus, ou tout lien sortant du dossier `--project`. L'IA ne peut jamais lire ni modifier un fichier en dehors du dossier cible.
- **Liste blanche Shell (`tools/shell.py`)** : L'outil `run_command` refuse toute commande qui ne commence pas par un préfixe autorisé dans `config/router.yaml` (`pytest`, `python`, `git status`, `git diff`, `ruff`, `ls`). De plus, tous les opérateurs d'enchaînement ou de redirection (`;`, `&&`, `||`, `|`, `` ` ``, `$(`, `>`, `<`) sont bloqués pour empêcher toute injection shell.

### 6.3 Plafond budgétaire (`CostTracker`)
- Le budget est vérifié **avant chaque appel au Planificateur** et **avant chaque tour de boucle ReAct de l'Exécuteur**.
- Si le coût cumulé atteint `--max-budget`, l'orchestration s'arrête immédiatement.
- *Note* : comme le nombre exact de tokens générés en sortie n'est connu qu'à la fin d'une requête HTTP, le tout dernier appel peut dépasser de quelques centimes le plafond fixé.

### 6.4 Comment rédiger un bon `--goal` ?
- Soyez explicite sur les **fichiers attendus** et le **moyen de vérification** :
  - *Moins bon* : `--goal "Fais un script de calcul"`
  - *Excellent* : `--goal "Créer calculator.py avec les fonctions add, sub, mul, div (gérant la division par zéro) et test_calculator.py vérifié avec pytest"`

---

## 7. Dépannage (FAQ)

- **`EnvironmentError: La variable d'environnement GEMINI_API_KEY est absente`**
  Vérifiez que votre fichier `.env` existe à la racine du dépôt et contient `GEMINI_API_KEY=...` (ou exportez la variable dans votre terminal : `$env:GEMINI_API_KEY="votre_cle"`).
- **`PermissionError: Commande refusée par la liste blanche`**
  L'Exécuteur a tenté de lancer une commande shell non listée dans `config/router.yaml` (par exemple `pip install` ou `mkdir`). Si vous souhaitez autoriser une commande supplémentaire, ajoutez son préfixe dans `config/router.yaml` sous `shell.allowed_commands`.
- **`MaxReactTurnsExceeded` sur une étape**
  L'Exécuteur n'a pas réussi à terminer l'étape en 4 tours d'outils (valeur par défaut `max_react_turns=4`). Le `StepRouter` incrémente alors le compteur d'échecs et, au bout de 2 échecs, escalade automatiquement au Planificateur pour découper l'étape plus finement.
