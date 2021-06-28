-- Licensed to the Apache Software Foundation (ASF) under one or more
-- contributor license agreements.  See the NOTICE file distributed with
-- this work for additional information regarding copyright ownership.
-- The ASF licenses this file to You under the Apache License, Version 2.0
-- (the "License"); you may not use this file except in compliance with
-- the License.  You may obtain a copy of the License at
--    http://www.apache.org/licenses/LICENSE-2.0
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.


DROP TABLE IF EXISTS `alembic_version`;
CREATE TABLE `alembic_version` (
	`version_num` VARCHAR(32) NOT NULL,
	CONSTRAINT `alembic_version_pkc` PRIMARY KEY (`version_num`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

DROP TABLE IF EXISTS `experiments`;
CREATE TABLE `experiments` (
	`experiment_id` INTEGER NOT NULL,
	`name` VARCHAR(256) NOT NULL,
	`artifact_location` VARCHAR(256),
	`lifecycle_stage` VARCHAR(32),
	CONSTRAINT `experiment_pk` PRIMARY KEY (`experiment_id`),
	UNIQUE (`name`),
	CONSTRAINT `experiments_lifecycle_stage` CHECK (`lifecycle_stage` IN ('active', 'deleted'))
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

DROP TABLE IF EXISTS `registered_models`;
CREATE TABLE `registered_models` (
	`name` VARCHAR(256) NOT NULL,
	`creation_time` BIGINT,
	`last_updated_time` BIGINT,
	`description` VARCHAR(5000),
	CONSTRAINT `registered_model_pk` PRIMARY KEY (`name`),
	UNIQUE (`name`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

DROP TABLE IF EXISTS `experiment_tags`;
CREATE TABLE `experiment_tags` (
	`key` VARCHAR(250) NOT NULL,
	`value` VARCHAR(5000),
	`experiment_id` INTEGER NOT NULL,
	CONSTRAINT `experiment_tag_pk` PRIMARY KEY (`key`, `experiment_id`),
	FOREIGN KEY(`experiment_id`) REFERENCES `experiments` (`experiment_id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

DROP TABLE IF EXISTS `model_versions`;
CREATE TABLE `model_versions` (
	`name` VARCHAR(256) NOT NULL,
	`version` INTEGER NOT NULL,
	`creation_time` BIGINT,
	`last_updated_time` BIGINT,
	`description` VARCHAR(5000),
	`user_id` VARCHAR(256),
	`current_stage` VARCHAR(20),
	`source` VARCHAR(500),
	`run_id` VARCHAR(32),
	`status` VARCHAR(20),
	`status_message` VARCHAR(500),
	`run_link` VARCHAR(500),
	CONSTRAINT `model_version_pk` PRIMARY KEY (`name`, `version`),
	FOREIGN KEY(`name`) REFERENCES `registered_models` (`name`) ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

DROP TABLE IF EXISTS `registered_model_tags`;
CREATE TABLE `registered_model_tags` (
	`key` VARCHAR(250) NOT NULL,
	`value` VARCHAR(5000),
	`name` VARCHAR(256) NOT NULL,
	CONSTRAINT `registered_model_tag_pk` PRIMARY KEY (`key`, `name`),
	FOREIGN KEY(`name`) REFERENCES `registered_models` (`name`) ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

DROP TABLE IF EXISTS `runs`;
CREATE TABLE runs (
	`run_uuid` VARCHAR(32) NOT NULL,
	`name` VARCHAR(250),
	`source_type` VARCHAR(20),
	`source_name` VARCHAR(500),
	`entry_point_name` VARCHAR(50),
	`user_id` VARCHAR(256),
	`status` VARCHAR(9),
	`start_time` BIGINT,
	`end_time` BIGINT,
	`source_version` VARCHAR(50),
	`lifecycle_stage` VARCHAR(20),
	`artifact_uri` VARCHAR(200),
	`experiment_id` INTEGER,
	CONSTRAINT `run_pk` PRIMARY KEY (`run_uuid`),
	FOREIGN KEY(`experiment_id`) REFERENCES `experiments` (`experiment_id`),
	CONSTRAINT `source_type` CHECK (`source_type` IN ('NOTEBOOK', 'JOB', 'LOCAL', 'UNKNOWN', 'PROJECT')),
	CONSTRAINT `runs_lifecycle_stage` CHECK (`lifecycle_stage` IN ('active', 'deleted')),
	CHECK (status IN ('SCHEDULED', 'FAILED', 'FINISHED', 'RUNNING', 'KILLED'))
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

DROP TABLE IF EXISTS `latest_metrics`;
CREATE TABLE `latest_metrics` (
	`key` VARCHAR(250) NOT NULL,
	`value` FLOAT NOT NULL,
	`timestamp` BIGINT,
	`step` BIGINT NOT NULL,
	`is_nan` BOOLEAN NOT NULL,
	`run_uuid` VARCHAR(32) NOT NULL,
	CONSTRAINT `latest_metric_pk` PRIMARY KEY (`key`, `run_uuid`),
	FOREIGN KEY(`run_uuid`) REFERENCES `runs` (`run_uuid`),
	CHECK (`is_nan` IN (0, 1))
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

DROP TABLE IF EXISTS `metrics`;
CREATE TABLE `metrics` (
	`key` VARCHAR(250) NOT NULL,
	`value` FLOAT NOT NULL,
	`timestamp` BIGINT NOT NULL,
	`run_uuid` VARCHAR(32) NOT NULL,
	`step` BIGINT DEFAULT '0' NOT NULL,
	`is_nan` BOOLEAN DEFAULT '0' NOT NULL,
	CONSTRAINT `metric_pk` PRIMARY KEY (`key`, `timestamp`, `step`, `run_uuid`, `value`, `is_nan`),
	FOREIGN KEY(`run_uuid`) REFERENCES `runs` (`run_uuid`),
	CHECK (`is_nan` IN (0, 1))
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

DROP TABLE IF EXISTS `model_version_tags`;
CREATE TABLE `model_version_tags` (
	`key` VARCHAR(250) NOT NULL,
	`value` VARCHAR(5000),
	`name` VARCHAR(256) NOT NULL,
	`version` INTEGER NOT NULL,
	CONSTRAINT `model_version_tag_pk` PRIMARY KEY (`key`, `name`, `version`),
	FOREIGN KEY(`name`, version) REFERENCES `model_versions` (`name`, `version`) ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

DROP TABLE IF EXISTS `params`;
CREATE TABLE `params` (
	`key` VARCHAR(250) NOT NULL,
	`value` VARCHAR(250) NOT NULL,
	`run_uuid` VARCHAR(32) NOT NULL,
	CONSTRAINT `param_pk` PRIMARY KEY (`key`, `run_uuid`),
	FOREIGN KEY(`run_uuid`) REFERENCES runs (`run_uuid`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

DROP TABLE IF EXISTS `tags`;
CREATE TABLE `tags` (
	`key` VARCHAR(250) NOT NULL,
	`value` VARCHAR(5000),
	`run_uuid` VARCHAR(32) NOT NULL,
	CONSTRAINT `tag_pk` PRIMARY KEY (`key`, `run_uuid`),
	FOREIGN KEY(`run_uuid`) REFERENCES `runs` (`run_uuid`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
