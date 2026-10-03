# LPWM-FM B full130 × 80k — 2026-09-22

User requests audit/archive completed common40 results, then same model settings on allLIBERO including90,
80kupdates. Previous40run remains immutable. Freshinitialization (no implicit30kcheckpointresume), same
initialweightSHA3116e036247b84884b6c259de7bf5a5fbdfa6737a8e763c8a2157f5c01d2e13e.

Scope: Spatial10/Object10/Goal10/LIBERO90/LIBERO10 =130taskinstances,112uniqueinstructions. This is
jointtraining, NOT LIBERO90-to10heldouttaskgeneralization. Use all50officialdemonstrations/task,6500episodes,
alloriginalrows; no sourcefiltering/resampling. Existingcommon40convertedpackage1693episodes is not silently
mixed withofficialraw. Old50.75%common40score selects chronology,not a matched full130baseline.

Source: pinned official catalog already audited2026-09-21,
yifengzhu-hf/LIBERO-datasets@f13aa24a3da8c43c7225569f28c562979fa0e35a.
Mirror istransportonly; eachHDF5size/SHAchecked. Allimageblocks exactroundtrip XOR16/zlibRGB128; rawOpenGL
rotate180once matchingruntime. Preserve native action7 / eefstate8, allhistory/actiontargets episode-safe.
Scene/expert mustnotsee futureobservations. Train-onlystate normalization; identityactions. Frozenreal
SmolVLMembeddings, exacttextduplicates sharetokens, no taskIDfeatures. Per-taskseed42random10%episodeholdout.

UnchangedBmodel/optimization: shared nativeDLP, scene2/expert4/world4width256, ordinaryFM(notIMF), language
andproprio, GTclean actiondynamics, conditionactiontokenrepeat3(noAdaLN); no predictedactionworldloss or
latentactionalignment. World1/rec1/dyn1/prior.001; worldramp1000. RGB128two cameras,history2,horizon16,
queue8,Euler10. Physicalbatch8×accum4effective32,float32/TF32. L40S16/32physicalbatchpreviouslyOOM.
AdamW1e-4,500warmup,cosineendpoint80000→1e-5,gradclip10,seed42. ExtendONLYbudget+tasks+decayendpoint.
OriginalSpatial30ksweep CLI stillrejects80k; fullscope explicitlyauthorizes --schedule-steps80000.
Savedoptimizer schedulemetadata is80000, includingpreflight.

Offlinevalidationevery500updates,130batches×8=1040balancedwindows, actualsimcheckpointgates every5000:
16checkpoints×130tasks×10episodes =20800repeatedvalidationrollouts. Fullnativetaskhorizons/no truncation,
per-task/per-suite/aggregate success, taskweightedandmacroresults. Evaluation pausestraining onsingleGPU.
Preserveall16periodiccheckpoints; best_loss aliasesbestamongperiodics (not every500steps). No optimistic
resumefromweights; process/disk/eval/uploaderror stops pipeline.80krequiredtofinishbeforeCOMPLETED.
PrivateonlineSwanLab projectlpwm-fm-b-full-libero; runfull130-B-w1-rec1-dyn1-cos80k-seed42.

Storage: root/root/gpufree-data/lpwm_full80/20260922. Reuse62officialtaskcaches byverifiedhardlinks, nevermodify
oldcompletedfiles. Newimages distributebetweendataplane(primarynewlimit9.5GiB) andexplicitoverflowroot
/root/lpwm_full80_overflow/20260922/cache. Whitelistedsymlinkroots+allfileSHAchecked; allotherescapesrefused.
Rawdownloadscratch usesoppositefilesystem, data-tierpairworkers/system-tierserial toboundspace. Downloads
removedonlyafternewcachecommit. No existingcache/checkpoint/log deletion. Systemoverflowloss causesfailclosed
read/integrityfailure, notsilentsubstitution. Needkeepmachine/systemdiskuntilruncompletion.
Finalfree>=5GiBdatadiskgate beforetraining,~3.5GiBfor16weights+optimizer; monitorremainingcapacity. Existing
supervisordata-prepguardfor80k=3GiB accommodatesboundednewrawscratch, outputcacheguard5.5GiBdata/2.5GiBsystem.

Preflight: actualnativeGPUtrainingandfive2-stepenvchecks; fullworldstrength memoryprobe. Preflightresults
NOT successrates. Confirm full130manifest+taskcoverage+allcache/sourcehashes+80kpreflightbeforeformalstart.

## User storage revision, 2026-09-22 (supersedes no-deletion policy above)

During preparation the user explicitly authorized retiring overlapping old datasets to avoid wasted space.
After exact local-backup SHA checks and no-open-FD/map verification, retired old common40 compressed image
payloads, the legacy Spatial image array, and four abandoned source-download partials. Deleted16397447619bytes
(~15.27GiB), no checkpoint/log/scalar/index/language deletion. Cache RETIRED_IMAGE_PAYLOADS markers document
local restoration sources. Common40converted andofficial130 RGB are not asserted bitwiseidentical; only the
retiredremote-vs-localbackup copies are exact. The reused62official caches alreadysharehardlinkedblocks.

The sourcecatalog itself has130distinctHDF5SHA256values: no exactduplicate sourcefiles among130tasks. Its
112uniqueinstructionstrings doNOTauthorize removingdifferent tasks/demonstrations withsame language.

After freeing capacity, preparation resumed all107completedtasks and raised primary-new-limit9.5→24GiB,
placing remainingnewtasks preferentiallyonthedata volume. One alreadycreated overflowtask remains whitelisted;
no model/seed/loss/schedule/eval contract changed. Fullrecord at runrootduplicate-cache-cleanup.json,
source-duplicate-audit.json andstorage-plan-revision.json. Original frozenpre-revisionplan remains historical.
