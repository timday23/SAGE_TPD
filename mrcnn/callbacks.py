import os
import numpy as np
import tensorflow as tf
import yaml
from tensorflow import keras
from mrcnn import utils
from mrcnn.model import load_image_gt, mold_image
import csv
import time


##TODO - Add progress bars for mAP - try to match keras progress bars?
class MeanAveragePrecisionCallback(keras.callbacks.Callback):
    def __init__(self, train_model, inference_model, dataset,
                 calculate_map_at_every_X_epoch=5, dataset_limit=None,
                 log_dir=None, verbose=1):
        
        super().__init__()
        self.train_model = train_model
        self.inference_model = inference_model
        self.dataset = dataset
        self.calculate_map_at_every_X_epoch = calculate_map_at_every_X_epoch
        self.dataset_limit = len(self.dataset.image_ids)
        if dataset_limit is not None:
            self.dataset_limit = dataset_limit
        self.dataset_image_ids = self.dataset.image_ids.copy()

        if inference_model.config.BATCH_SIZE != 1:
            raise ValueError("This callback only works with the bacth size of 1")

        self.log_dir = log_dir or train_model.log_dir
        self._verbose_print = print if verbose > 0 else lambda *a, **k: None

        #tensorboard logging
        self.file_writer = tf.summary.create_file_writer(train_model.log_dir)

        # Flag to check if inference model is built
        self._inference_model_built = False


    def on_epoch_end(self, epoch, logs=None):
        
        #if epoch > 2 and (epoch+1)%self.calculate_map_at_every_X_epoch == 0:
        if (epoch+1) >= 1 and (epoch+1) %self.calculate_map_at_every_X_epoch == 0: #can i change it to start at epoch 1?
            self._verbose_print("Calculating mAP...")
            self._load_weights_for_model()

            AP50s, mAPs_COCO = self._calculate_mean_average_precision()
            mean_AP50 = np.mean(AP50s)
            mAP_COCO = np.mean(mAPs_COCO)
            #update logs for Keras callbacks
            if logs is not None:
                self._verbose_print("Updating logs with mAP results")
                logs["val_AP50"] = mean_AP50
                logs["val_mAP_coco"] = mAP_COCO
            else:
                self._verbose_print("No logs provided, skipping update")

            self._verbose_print("mean AP50 at epoch {0} is: {1}".format(epoch+1, mean_AP50))
            self._verbose_print("mAP COCO at epoch {0} is: {1}".format(epoch+1, mAP_COCO))

            #tensorboard writer (doesnt work)
            if self.file_writer is not None:
                with self.file_writer.as_default():
                    tf.summary.scalar("val_AP50", mean_AP50, step=epoch+1)
                    tf.summary.scalar("val_mAP_coco", mAP_COCO, step=epoch+1)
                    self.file_writer.flush()
            

        super().on_epoch_end(epoch, logs)

    def _ensure_model_built(self, model, config):
        """
        Ensure that all dynamic heads (e.g., PP head with GT clusters) are fully built
        so that Keras layers have fixed shapes and weights can be loaded.
        """
        # If the model has been called before, skip
        if getattr(model, "_built_with_dummy_input", False):
            return

        # Create dummy inputs with batch_size=1
        batch_size = 1
        h, w, c = config.IMAGE_SHAPE
        dummy_image = np.zeros((batch_size, h, w, c), dtype=np.float32)
        dummy_meta = np.zeros((batch_size, config.IMAGE_META_SIZE), dtype=np.float32)
        dummy_anchors = np.zeros((batch_size, config.POST_NMS_ROIS_INFERENCE, 4), dtype=np.float32)

        inputs = [dummy_image, dummy_meta, dummy_anchors]

        if config.USE_PP_HEAD and config.PP_USE_GT_CLUSTERS:
            dummy_clusters = np.zeros((batch_size, h, w, config.MAX_CLUSTERS), dtype=np.float32)
            inputs.append(dummy_clusters)

        # Do a single forward pass to finalize all layer shapes
        model.keras_model.predict(inputs)

        # Mark that we already did the dummy call
        model._built_with_dummy_input = True
    
    # def _load_weights_for_model(self):
    #     last_weights_path = self.train_model.find_last_in_folder(self.log_dir)
    #     if not last_weights_path or not os.path.exists(last_weights_path):
    #         raise FileNotFoundError(f"No checkpoint found in {self.log_dir}")
    #     self._verbose_print("Loaded weights for the inference model (last checkpoint of the train model): {0}".format(
    #         last_weights_path))
    #     self.inference_model.load_weights(last_weights_path,
    #                                       by_name=True)
        # Make sure model is fully built first
        #self._ensure_model_built(self.inference_model, self.inference_model.config)

        # Then load weights safely
        #self.inference_model.keras_model.set_weights(self.train_model.keras_model.get_weights())
        #self.inference_model.keras_model.set_weights(self.train_model.keras_model.get_weights())
        #self._verbose_print("Copied weights from training model to inference model for mAP evaluation,")

    # def _load_weights_for_model(self):
    #     """
    #     Load weights from disk and ensure the model is ready for mAP evaluation.
    #     """
    #     # Find last checkpoint
    #     last_weights_path = self.train_model.find_last_in_folder(self.log_dir)
    #     if not last_weights_path or not os.path.exists(last_weights_path):
    #         raise FileNotFoundError(f"No checkpoint found in {self.log_dir}")
    #     self._verbose_print(f"Loading weights for inference model: {last_weights_path}")

    #     # Load weights safely
    #     self.inference_model.load_weights(last_weights_path, by_name=True)

    #     # Mark model as built
    #     self.inference_model._built_with_dummy_input = True

    def _load_weights_for_model(self):
        last_weights_path = self.train_model.find_last_in_folder(self.log_dir)
        if not last_weights_path or not os.path.exists(last_weights_path):
            raise FileNotFoundError(f"No Checkpoint found in {self.log_dir}")
        
        # self._verbose_print("\n===== WEIGHT LOADING DEBUG =====")
        # self._verbose_print(f"Checkpoint path: {last_weights_path}")

        config = self.inference_model.config

        self._verbose_print("Inference model expects inputs:")
        # for inp in self.inference_model.keras_model.inputs:
        #     self._verbose_print(f"  {inp.name}  shape={inp.shape}")

        from mrcnn import model as modellib

        #build fresh inference model:
        new_model = modellib.MaskRCNN(
            mode="inference",
            config=config,
            model_dir=self.log_dir
        )

        # Load weights into correctly-shaped graph
        new_model.load_weights(last_weights_path, by_name=True)
        self.inference_model = new_model
        self._verbose_print("Inference model rebuilt and weights loaded cleanly.")

    def _calculate_mean_average_precision(self):
        AP50s = []
        mAPs_COCO = []
        # Use a random subset of the data when a limit is defined
        np.random.shuffle(self.dataset_image_ids)

        dataset = self.dataset
        config = self.inference_model.config

        for image_id in self.dataset_image_ids[:self.dataset_limit]:
            image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, config,
                                                                             image_id)
            

            #molded_images = np.expand_dims(mold_image(image, self.inference_model.config), 0)
            # Compute effective scale (same as before)
            # info = dataset.image_info[image_id]
            # original_height = info.get("height", image.shape[0])
            # original_width = info.get("width", image.shape[1])
            # original_scale = info.get("scale", 1.0)
            # resized_height, resized_width = molded_images.shape[:2]
            # effective_scale = original_scale * (resized_height / original_height)

            # GT clusters (if used) - resize as before
            gt_clusters = None
            if getattr(dataset, "load_cluster_as_input", False) and getattr(config, "PP_USE_GT_CLUSTERS", False):
                gt_clusters = dataset.load_cluster_input_masks(image_id)
                if gt_clusters.size != 0:
                    _, window, scale, padding, crop = utils.resize_image(
                        dataset.load_image(image_id),
                        min_dim=config.IMAGE_MIN_DIM,
                        min_scale=config.IMAGE_MIN_SCALE,
                        max_dim=config.IMAGE_MAX_DIM,
                        mode=config.IMAGE_RESIZE_MODE,
                    )
                    gt_clusters = utils.resize_mask(gt_clusters, scale, padding, crop)
                else:
                    gt_clusters = None

            results = self.inference_model.detect([image],
                              gt_cluster_masks=[gt_clusters] if gt_clusters is not None else None,
                               verbose=0)
            #results = self.inference_model.detect(molded_images, verbose=0)
            r = results[0]
            # Compute mAP - VOC uses IoU 0.5
            AP, _, _, _ = utils.compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"],
                                           r["class_ids"], r["scores"], r['masks'])
            AP50s.append(AP)

            mAP_COCO = utils.compute_ap_range(gt_bbox, gt_class_id, gt_mask,
                                              r["rois"], r["class_ids"], r["scores"], r['masks'],
                                              verbose=0)
            mAPs_COCO.append(mAP_COCO)



        return np.array(AP50s), np.array(mAPs_COCO)
    


class TrainingLogger(keras.callbacks.Callback):

    def __init__(self, train_model, log_dir=None, dataset_train=None,dataset_val=None,
                 init_with = "coco", manual_weights_path=None, 
                 scheduled_epochs=None, stage_name=None, metric_log_cats = None, verbose=1):
        
        """
        model: MaskRCNN training model
        log_dir: directory to save logs and metrics file, i.e. same as where checkpoints are saved
        dataset_train, dataset_val: dataset objects for training and validation
        init_with: str, one of "coco", "last", "manual" or "imagenet"
        manual_weights_path: if init_with is "manual", path to weights file
        """
        super().__init__()
        self.train_model = train_model
        self.log_dir = log_dir or train_model.log_dir
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.init_with = init_with
        self.manual_weights_path = manual_weights_path
        self.verbose = verbose
        self.scheduled_epochs =scheduled_epochs
        self.stage_name = stage_name
        self.config = self.train_model.config
        self.epoch_start_time = None

        #training log YAML file path
        self.logfile_path = os.path.join(self.log_dir, "training_log.yaml")

        #metric log csv file path
        self.results_file = os.path.join(self.log_dir, "metric_log.csv")
        self._file_initialized = False

        #metric categories for csv logging
        self.metric_log_cats = metric_log_cats
        self.predefined_cats = {
            "mAP": ["val_AP50", "val_mAP_coco"],
            "loss": ["loss", "val_loss", "mrcnn_mask_loss", "val_mrcnn_mask_loss", "pp_coverage_loss","val_pp_coverage_loss"],
            "general": ["loss","val_loss",  "mrcnn_mask_loss", "val_mrcnn_mask_loss", "pp_coverage_loss","val_pp_coverage_loss","val_AP50", "val_mAP_coco"]
        }

        #internal storage
        self.log_data = {
            "model_dir": self.log_dir,
            "starting_weights": self.init_with,
            "manual_weights_path": self.manual_weights_path,
            #"config": {},   
            "datasets": {},
            "training_stages": [],
        }

        
        


        #save config and dataset info immediately
        self._save_full_config() #maybe move these to on_train_begin?
        self._save_initial_info()
        
        self.last_epoch=None
        self.best_epoch = None
        

    def _verbose_print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def _get_dataset_info(self, dataset):
        if not dataset:
            return None
        
        classes = [c.lower() for c in dataset.class_names]
        mask_info =  {}

        if "particle" in classes and getattr(dataset, "particle_masks_dir", None):
            mask_info["particle"] = dataset.particle_masks_dir

        if "cluster" in classes and getattr(dataset, "cluster_masks_dir", None):
            mask_info["cluster"] = dataset.cluster_masks_dir

        return {
            "classes": dataset.class_names,
            "mask_paths": mask_info,
            "num_images": len(dataset.image_ids)
            }

    def _save_initial_info(self):
        #ONLY WRITIE if log file does not exist
        if not os.path.exists(self.logfile_path):
            simple_config_dict = {
                k: v for k, v in vars(self.config).items()
                if isinstance(v, (int, float, str, list, tuple, bool, type(None))) 
            }

            self.log_data["config"] = simple_config_dict

            if self.dataset_train:
                
                self.log_data["datasets"]["train"] = self._get_dataset_info(self.dataset_train)
                
                #{
                #    "path": getattr(self.dataset_train, "particle_masks_dir",None),
                #    "num_images": len(self.dataset_train.image_ids)
                #}

            if self.dataset_val:
                self.log_data["datasets"]["val"] = self._get_dataset_info(self.dataset_val)
                
                #{
                #    "path": getattr(self.dataset_val, "particle_masks_dir",None),
                #    "num_images": len(self.dataset_val.image_ids)
                #}

            self._write_yaml()
            self._verbose_print(f"Training log saved to {self.logfile_path}")


    def _get_metrics_for_epoch(self, logs):
        logs = logs or {}
        if self.metric_log_cats == "all":
            #log all keys in logs 

            keys = list(logs.keys())
        else:
            keys = self.predefined_cats.get(self.metric_log_cats,[])
        values = []
        for key in keys:
            val = logs.get(key, "")
            values.append(round(float(val),4) if val != "" else "")
        return keys, values
        

    
    def _save_full_config(self):
        import yaml
        full_config_dict = {k: getattr(self.config, k) 
                            for k in dir(self.config) 
                            if not k.startswith('_') and not callable(getattr(self.config, k))}
        
        #add training schedule if it exists
        if hasattr(self.config, "TRAIN_SCHEDULE"):
            full_config_dict["TRAIN_SCHEDULE"] = self.config.TRAIN_SCHEDULE

        # ensure directory exists
        os.makedirs(self.log_dir, exist_ok=True)
        

        # YAML
        yaml_path = os.path.join(self.log_dir, "config.yaml")
        try:
            with open(yaml_path, "w") as f:
                yaml.dump(full_config_dict, f, sort_keys=False)
            self._verbose_print(f"Full config saved to YAML: {yaml_path}")
        except Exception as e:
            self._verbose_print(f"[ERROR] Failed to save YAML: {e}")

        # TXT
        txt_path = os.path.join(self.log_dir, "config.txt")
        try:
            with open(txt_path, "w") as f:
                for k, v in full_config_dict.items():
                    f.write(f"{k}: {v}\n")
            self._verbose_print(f"Full config saved to TXT: {txt_path}")
        except Exception as e:
            self._verbose_print(f"[ERROR] Failed to save TXT: {e}")


    def _write_yaml(self):
        import yaml
        os.makedirs(os.path.dirname(self.logfile_path), exist_ok=True) 
        with open(self.logfile_path, 'w') as f:
            yaml.dump(self.log_data, f, sort_keys=False)

    
 
    # ------Callback methods------        

    def on_train_begin(self, logs=None):
        if self.stage_name is None:
            self.stage_name = f"stage_{len(self.log_data['training_stages'])+1}"
        #store timestamp for start time
        self.stage_start_time = time.time()
        self.stage_epoch_times_total = 0.0
        self.stage_epochs_completed = 0

        #determine csv headers based on metric category
        if self.metric_log_cats == "all":
            self.csv_keys=None
        else:
            self.csv_keys = self.predefined_cats.get(self.metric_log_cats, [])



        #if csv doesnt exist, create it with headers
        if not os.path.exists(self.results_file):
            with open(self.results_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                headers = ["epoch", "stage"]
                headers += self.csv_keys if self.csv_keys is not None else ["metrics_placeholder"]
                headers.append("epoch_time")
                writer.writerow(headers)

        self._file_initialized = True

            
        self._verbose_print(f"Training started for '{self.stage_name}' layers "
                            f"({self.scheduled_epochs} scheduled epochs). Log File: {self.logfile_path}")
        
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
                
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        #---timing----
        epoch_time = time.time() - self.epoch_start_time
        self.stage_epoch_times_total += epoch_time
        self.stage_epochs_completed +=1

        avg_epoch_time = self.stage_epoch_times_total / self.stage_epochs_completed
        stage_total_time = time.time() - self.stage_start_time

        #---get current learning rate
        lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        # get metrics for csv logging and storage
        metric_keys, metric_values = self._get_metrics_for_epoch(logs)
        
        

        #---- epoch metrics----
        epoch_entry = {
            "epoch": int(epoch+1),
            "learning_rate": round(lr, 6),
            "losses": {k: round(float(v), 4) for k,v in logs.items() if "loss" in k},
            "metrics": {k: round(float(v),4) for k,v in logs.items() if "AP" in k or "mAP" in k},
            "epoch_time": format_time(epoch_time) #convert to HH:MM:SS
        }

        self.last_epoch = epoch_entry

        #----update best epoch based on val_loss
        best_loss = self.best_epoch["losses"].get("val_loss", float("inf")) if self.best_epoch else float("inf")
        current_loss = epoch_entry["losses"].get("val_loss", float("inf"))
        if current_loss < best_loss:
            self.best_epoch = epoch_entry
        
        
        stage_summary = {
            "stage": self.stage_name,
            "scheduled_epochs": self.scheduled_epochs,
            "actual_epochs": epoch + 1,
            "last_epoch": epoch_entry,
            "best_epoch": self.best_epoch ,
            "stage_total_time": format_time(stage_total_time), #convert to minutes if over 3 mins, hours if over 2 hours
            "stage_avg_epoch_time": format_time(avg_epoch_time), #convert to minutes if over 3 mins, hours if over 2 hours
        }
        #replace previous stage summary if exists
        self.log_data["training_stages"] = [
            s for s in self.log_data["training_stages"] if s.get("stage") != self.stage_name
        ]
        self.log_data["training_stages"].append(stage_summary)
        self._write_yaml()

        #csv logging
        if self._file_initialized:
            # get keys and values based on metric category
            #if self.metric_log_cats == "all":
            #    keys = list(logs.keys())
            #else:
            #    keys = self.predefined_cats.get(self.metric_log_cats, [])
            #values = [round(float(logs.get(k, "")), 4) if logs.get(k) is not None else "" for k in keys]

            #include epoch time in both formats
            #keys += ["epoch_time"]
            #values += format_time(epoch_time)

            #val_AP50 = epoch_entry["metrics"].get("val_AP50","")
            #val_mAP_coco = epoch_entry["metrics"].get("val_mAP_coco","")

            #construct full row dict
            row = {"epoch": epoch+1, "stage": self.stage_name}
            row.update(dict(zip(metric_keys, metric_values)))
            row["epoch_time"] = format_time(epoch_time)

            #write row using header order
            header_order = ["epoch","stage"]+metric_keys+ ["epoch_time"]
            with open(self.results_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([row[h] for h in header_order])
        

def format_time(seconds: float) -> str:
    """Converts seconds to HH:MM:SS format"""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"