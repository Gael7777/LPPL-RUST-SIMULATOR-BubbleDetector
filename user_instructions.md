# Instructions utilisateur (simple et rapide)

## Prérequis
- Installer Rust (gratuit, 5 minutes) : https://rustup.rs/
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
   ```
   .\target\release\hlpll-gui.exe
   ```
   - Interface moderne avec sliders, graphs interactifs (prix, score bulle, équité 10k€)
   - Tu changes les paramètres en direct (ticker, dates, seuils, "Long only" / "Short only", etc.)
   - Boutons "Test Yahoo", "Run Simulation", "Export CSV/PNG"
   - Compare directement à Buy & Hold sur les graphs

5. **Version terminal interactive (TUI)** :
   ```
   cargo build --release --bin hlpll-explorer --no-default-features --features tui
   ```
   Puis :
   ```
   .\target\release\hlpll-explorer.exe
   ```
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
- Exemples de tickers : CAR, AMD, AMTX (actions volatiles qui bougent fort).

C'est tout ! Clone le repo, suis les commandes ci-dessus, et amuse-toi à tweaker les paramètres. Si tu as des questions sur les graphs ou les résultats, lis les CSV générés dans `results/`.