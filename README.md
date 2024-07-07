# T-AIA-902

> _Un projet de de recherche et de découverte du domaine du traitement de l'apprentissage renforcé (RL), de ses applications et de 
ses outils développés au fil des avancées de la recherche._

## Sommaire
- [Groupe](#groupe)
- [Le projet](#le-projet)
  - [Contexte](#contexte)
  - [Architecture](#architecture)
- [Rapport](#rapport)
- [Instructions](#instructions)

## Groupe

Le groupe ayant réalisé l'ensemble du travail de recherche et de développement est composé de 4 membres :
- Eliott CLAVIER
- Marius GLO
- Eli MORICEAU
- Clément MATHÉ
- Paul RIPAULT

## Le projet

### Contexte

La demande initiale du projet __T-AIA-902__ émane d'une volonté de concevoir une solution capable d'apprendre d'un environnement donné
à partir de plusieurs algorithmes pour pouvoir choisir un chemin optimal.

_A noter que l'environnement utilisé tout au long du projet est issu de l'API Gym réalisée par OpenAI, notre besoin réside dans l'utilisation et
l'optimisation d'algorithmes dans celui-ci._

Pour répondre à cette demande, notre approche s’est reposée sur une exploration approfondie des technologies de pointe en matière de RL. 
Cette exploration nous a conduits à examiner plus particulièrement les tâches d' __optimisation d'hyperparamètres__ d'algorithmes
pour répondre à notre besoin.


### Architecture

![Image](./architecture_globale.png)

L'architecture du système conçue intègre une interaction cli dans le back-end pour la sélection et récupération des informations provenants des modèles. 

Le back-end est développé en Python et repose sur l'api Gym qui initialise l'environnement du Taxi driver.

## Rapport

L'ensemble de notre travail de recherche et d'exploration des technologies de RL est détaillé dans le rapport de projet
disponible dans le fichier `rapport_rl.pdf`. 

Ce rapport traite d'abord de l'environnement Taxi Driver, de son fonctionnement et de pourquoi il est intéressant à utiliser pour nos recherches.

Une grande partie du rapport est dédiée à la présentation des résultats de nos modèles à savoir l'impact de l'utilisation d'algorithmes
différents (SARSA, MONTECARLO, QLEARNING...). 

Le rapport conclu sur les intérêts et résultats que nous avons obtenu après avoir tester les différents modèles dans notre environnement.

## Instructions

Pour lancer le projet, il est nécessaire de lancer le back-end.

L'ensemble des instructions pour lancer le projet sont disponibles dans le fichier `README.md` du dossier `taxi-driver`, contentant respectivement le code de notre application python.