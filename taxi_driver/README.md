# Backend
Le backend est une application Python qui va permettre de lancer un terminal de commande dans lequel l'utilisateur pourra intéragir avec l'environnement.

## Installation
Dans le répertoire racine du projet, exécutez la suite de commandes qui suit, en notant qu'il est recommandé d'utiliser [Python 3.12.4](https://www.python.org/downloads/release/python-3124/).

Pour faire fonctionner le projet, il vous faudra installer __poetry__ [selon votre configuration](https://python-poetry.org/docs/). La commande par défaut est la suivante :
```powershell
pipx install poetry
```

__La version vérifiée de poetry pour ce projet est la 1.8.3.__

Une fois poetry installé, il faut le mettre à jour pour activer les dépendances avec les commandes suivantes :
```powershell
poetry lock
```
Puis :
```powershell
poetry update
```

## Lancement
Une fois l'environnement virtuel configuré, exécutez les commandes suivantes pour lancer l'application :

Lancer le terminal CLI avec la commande suivante :
```powershell
python launch_user_mode.py 
```

## Résultats

Tous les résultats des traitements sont disponibles dans le dossier static à la racine du projet.