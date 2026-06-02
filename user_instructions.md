# Instructions utilisateur (simple et rapide)

## Prérequis

- Installer Rust (gratuit, 5 minutes) : [https://rustup.rs/](https://rustup.rs/)
- Ouvrir PowerShell ou Invite de commandes

## Étapes pour builder et lancer

1. Placer-toi dans le dossier du projet (celui qui contient le fichier `Cargo.toml`).
2. **Builder la version de base (CLI)** :
  ```
   cargo build --release
  ```
   (première fois : 2-5 minutes, les suivantes sont plus rapides)
3. **Lancer le programme principal** :
  Les .exe sont dans `target/release/`
  - Programme CLI classique :
    ```
    .\target\release\hlpll-backtester.exe --help
    ```
    Exemple d'utilisation :
    ```
    .\target\release\hlpll-backtester.exe --tickers CAR,AMD --start 2023-01-01 --end 2024-12-31
    ```
    → Ça télécharge les données, calcule tout et sauve les résultats dans `results/`
4. **Version avec belle interface graphique Windows (recommandée)** :
  ```
   cargo build --release --bin hlpll-gui --no-default-features --features gui
  ```
   Puis lancer :
  - Interface moderne avec sliders, graphs interactifs (prix, score bulle, équité 10k€)
  - Tu changes les paramètres en direct (ticker, dates, seuils, "Long only" / "Short only", etc.)
  - Boutons "Test Yahoo", "Run Simulation", "Export CSV/PNG"
  - Compare directement à Buy & Hold sur les graphs
5. **Version terminal interactive (TUI)** :
  ```
   cargo build --release --bin hlpll-explorer --no-default-features --features tui
  ```
   Puis :
   (touches : F pour fetch, R pour run, flèches pour naviguer, ? pour aide, q pour quitter)

## Comment ça marche exactement (très simple)

- C'est un **backtester** pour une stratégie mathématique de détection de "bulles" sur les actions (modèle LPPL + volume + sentiment).
- Tu donnes un ticker (ex: CAR, AMD), une période, des paramètres (taille de fenêtre, seuils de score, coûts de transaction...).
- Le programme :
  1. Télécharge les prix OHLCV depuis Yahoo Finance.
  2. Fit un modèle LPPL sur des fenêtres glissantes de données passées.
  3. Calcule un "bubble score" (combinaison de surévaluation LPPL + hype volume + sentiment).
  4. Applique une stratégie : score haut → Long, score bas → Short, sinon Flat (ou modes Long Only / Short Only / Invert selon tes réglages).
  5. Simule les trades avec un capital de départ (par défaut 10 000 $) + coûts.
  6. Affiche les graphs (prix coloré par position, score bulle avec seuils, courbe d'équité vs Buy & Hold) + stats (rendement, Sharpe, drawdown, nombre de trades).
- Résultats : CSV détaillés + PNG dans le dossier `results/`.
- Tu peux tweaker tous les paramètres en live dans le GUI/TUI et relancer instantanément les simus pour comparer.

**Astuces** :

- La première compilation est longue (télécharge les dépendances).
- Le dossier `target/` n'est pas à partager (il est généré automatiquement).
- Pour tester rapidement le GUI sur Windows : c'est l'option la plus sympa et visuelle.

## Explication du "random seed" (RNG seed pour les fits LPPL)

- Dans le modèle LPPL : ln(p_t) = A + B * tau^m + C * tau^m * cos(omega * ln(tau) + phi) où tau=tc-t.
- 4 params non-linéaires (tc, m, omega, phi) difficiles à optimiser (surface d'erreur non-convexe, plein de minima locaux).
- On fait du "multi-start random search" : on tire 1200+ combinaisons aléatoires dans des bornes raisonnables (via le RNG), pour chacune on résout analytiquement A/B/C par moindres carrés, on garde le meilleur fit valide (contraintes m, omega, damping).
- **Pourquoi du random ?** Pas de solver global lourd (choix "pure Rust simple" sans deps argmin etc). Le random restarts permet d'explorer globalement sans se coincer dans un mauvais minimum local.
- Le "random_seed" (défaut 42) fixe la séquence "aléatoire" : même données + même seed = exactement mêmes tirages = mêmes meilleurs params trouvés = backtest 100% reproductible (même scores, positions, equity, trades... chaque fois que tu cliques Run).
- Sans seed fixe (avant), c'était non-déterministe à chaque run (d'où les résultats différents).
- **On peut faire confiance pour du backtesting ?** Oui pour comparer des setups/stratégies de façon reproductible et équitable (même "hasard" contrôlé). Le search est large (1200 essais) + filtres de validité. Mais c'est une approx (pas optimum global garanti). Différents seeds donnent des modèles légèrement différents : utile pour tester robustesse.
- **Pour décider d'investir MAINTENANT ?** Non, ne te fie pas à ça seul. C'est un signal de recherche parmi d'autres (combine avec analyse fondamentale, news, risk management, etc.). Les backtests (même reproductibles) ne prédisent pas le futur. Ce n'est PAS un conseil financier. Toujours DYOR, diversifie, gère le risque. Le modèle peut aider à visualiser des régimes, mais l'investissement réel est risqué.

## Recommandations BUY/SELL/HOLD (via curseur dans les UIs)

- Dans GUI (hlpll-gui) : active "Track mouse hover", bouge la souris sur le graphique PRICE : le curseur suit en live, et en bas tu vois clairement "RECOMMENDATION AT [date]: BUY / GO LONG ..." ou SELL ou HOLD, avec le score et composants. Le slider permet aussi de naviguer précisément entre les dates.
- Dans Explorer TUI (hlpll-explorer) : utilise j/k pour déplacer le curseur clavier, la box "Live Bubble Indicator + RECOMMENDATION" affiche en gros le BUY/SELL/HOLD basé sur la position finale du signal à cette date (après seuils + bias/invert).
- Dans backtester CLI : lance avec --query-date 2024-05-31 pour avoir la reco précise à cette date dans la plage. Le summary inclut toujours la reco finale à la dernière date.
- La "sentiment" vient directement de la position calculée : >0 = BUY/LONG (score haut = momentum haussier selon le modèle), <0 = SELL/SHORT, =0 = HOLD/FLAT. C'est le résultat de la stratégie stricte à cette date (pas du prix seul).

Utilise le curseur pour "zoomer" sur n'importe quelle date dans tes limites et voir l'avis du modèle à ce moment précis. Amuse-toi, mais rappelle-toi : backtest != garantie future !

- Exemples de tickers : CAR, AMD, AMTX (actions volatiles qui bougent fort).

## Nouvelles fonctionnalités étendues (2026) : prédiction future de bulles + trading live sur sentiment actuel (LPPLS C1)

Le projet a été mis à jour de façon très complète après lecture de `grok-build-assets/gemini-data-LPPLS.md` (modèle LPPLS amélioré, indice de confiance C1 = % de fenêtres valides avec filtres stricts JLS, projection tc dates futures, niveaux de risque).

**Run modes** (GUI : radios "Historical / Prediction / Live / Hybrid" ; CLI --mode ; TUI via champs) :
- historical : backtest classique + courbe equity 10k vs B&H (comme avant).
- prediction : prédiction future de bulle (C1 % confiance, niveau risque LOW/HIGH/CRITICAL, dates médianes de "pic critique" tc prédites par les fits valides, probabilité dans horizon).
- live : snapshot "sentiment actuel" pour trader maintenant (reco BUY/SELL/HOLD qui tient compte de tout + note actionable sur le risque C1 + tc médian si cluster).
- hybrid : les deux (valide la règle live sur l'historique + donne le snapshot actuel).

**Autres options puissantes** :
- --ensemble-seeds "42,43,44" (ou GUI/TUI) : moyenne de C1 sur plusieurs seeds → plus robuste.
- Filtres JLS (m 0.1-0.9, omega 4.5-13, B<0, tc offset) pour le C1 "officiel".
- use conf for flat / sizing : si C1 haut → force flat ou scale la taille de position (risk management).
- predict-horizon : pour calculer "proba tc dans les N jours".

**Exemples CLI** :
```
.\target\release\hlpll-backtester.exe --tickers CAR --mode prediction --ensemble-seeds "42,43" --predict-horizon 90
.\target\release\hlpll-backtester.exe --tickers AMD --mode live --invert --use-conf-flat
```

Dans GUI : choisis le mode, remplis ensemble/horizon/filtres, Run Simulation → le panneau en haut affiche C1 + tc médian + reco live si pertinent. Les graphs restent pour les runs historical/hybrid.

**Interprétation rapide (d'après la doc gemini + littérature)** : C1 >75% = CRITICAL (herd extrême, pic probable proche) ; 45-75 HIGH ; etc. Utilise pour alerte risque ou pour shorter les bulles (avec invert + shortonly). Toujours croiser avec autres signaux. Ce n'est PAS un conseil financier — outil de recherche extensif et reproductible (seed fixe).

C'est tout ! Clone le repo, suis les commandes ci-dessus, et amuse-toi à tweaker les paramètres. Si tu as des questions sur les graphs ou les résultats, lis les CSV générés dans `results/`. (Et le gros README.md pour la théorie + refs.)