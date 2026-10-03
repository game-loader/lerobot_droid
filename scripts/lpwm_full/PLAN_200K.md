# GB200 paired fresh200k runs — 2026-09-25

User authorized restarting both prior tasks FROM SCRATCH, each200k updates, each task8 evaluation.
This is NOT an80k→200k continuation and NOT initialization from a selected oldcheckpoint.

GPU0 full130: Spatial10/Object10/Goal10/LIBERO90/LIBERO10,130taskinstances.
GPU1 common40: Spatial10/Object10/Goal10/LIBERO10,40taskinstances.
Both use existing audited official data/caches, all50demos/task with the same seed42per-task10%episode validation split.
No redownload, image duplication, oldrun/source/checkpoint deletion or overwrite. Newrunroots are exclusive.

Same LPWM-FM B: native shared DLP, ordinary FMexpert, GTclean-action dynamics, repeated3conditiontokens,
noAdaLN/noIMF/noactionlatentalignment; scene2/expert4/world4,width256,8heads,world/rec/dyn1/1/1,prior.001,
worldramp1000. RGB128two cameras,history2,horizon16,execution8,Euler10,state8/action7. Exact frozen language
embeddingcache and per-scope train-onlystate normalization. Initialweights SHA256 must equal prior native
GB200 initialization ea7fa2f61a64c37332c9bf1778b4077be0d0ab4f2174a9142210b230c9ea6840.

Both physicalbatch32,gradaccumulation1,AdamWpeak1e-4,min1e-5,500warmup,200000cosineendpoint,gradclip10,
float32/TF32,seed42. Offlinevalidationevery500updates;130batches forfull130,40forcommon40 as before.
Checkpoints every5000,ALL40periodicexports perrun retained; best_loss aliases bestvalidationFM amongsaved
checkpoints only, not every500step minimum. No rollingthreecheckpoint pruning.

Checkpoint eval pauses its training:8spawnworkers perGPU,one entiretask perworker,10sequentialepisodes/task
with originalhardreset history,originalseeds/initstates/fullcontrolhorizons. Thus1300episodes/full130eval,
400/common40eval;40checkpoints→52000and16000repeatedvalidationepisodes respectively. No globalepisodequeue,
noWarp, no short-horizon formalresults. Canonicaltaskordering andsimulator-defined success stay unchanged.
Full130training includes LIBERO10; this is not90→10heldouttaskgeneralization. Report shared40-task scores
separately for comparisons; full130aggregate andcommon40aggregate have different denominators/tasksets.
One seed percondition is an exploratory run, not independent-seed statistical replication.

PerGPU full-world-strength batch32profile and fresh2-update actual-cache preflight must pass before formalstart;
preflight scheduler endpoint200k, realnative8process/suite smoke and privateSwanLabreadback. Two cards retain
independent UUID/EGL masks; no globalMPS changes. Sourcehashes/cachehashes/GPUownership/diskreserve checks
precedeproduction. Require at least30GiB combined free for both40-checkpoint runs, supervisor12GiB/run.

Production privateSwanLab project lpwm-fm-b-full-libero, independentnames forfull130/common40/200k/seed42.
Keeporiginal80kresults and link the newexperiment provenance, but do not splice oldmetrics into freshcurves.
Schedule endpoints differ from80kruns, so200k's first80kupdates are not an identical80k control.
