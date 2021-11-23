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

DROP TABLE IF EXISTS `registered_model`;
CREATE TABLE `registered_model` (
	`name` VARCHAR(256) NOT NULL,
	`creation_time` DATETIME(3) COMMENT 'Millisecond precision',
	`last_updated_time` DATETIME(3) COMMENT 'Millisecond precision',
	`description` VARCHAR(5000),
	CONSTRAINT `registered_model_pk` PRIMARY KEY (`name`),
	UNIQUE (`name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

DROP TABLE IF EXISTS `registered_model_tag`;
CREATE TABLE `registered_model_tag` (
	`name` VARCHAR(256) NOT NULL,
	`tag` VARCHAR(256) NOT NULL,
	CONSTRAINT `registered_model_tag_pk` PRIMARY KEY (`name`, `tag`),
	FOREIGN KEY(`name`) REFERENCES `registered_model` (`name`) ON UPDATE CASCADE ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

DROP TABLE IF EXISTS `model_version`;
CREATE TABLE `model_version` (
	`name` VARCHAR(256) NOT NULL COMMENT 'Name of model',
	`version` INTEGER NOT NULL,
	`source` VARCHAR(512) NOT NULL COMMENT 'Model saved link',
	`user_id` VARCHAR(64) NOT NULL COMMENT 'Id of the created user',
	`experiment_id` VARCHAR(64) NOT NULL,
	`model_type` VARCHAR(64) NOT NULL COMMENT 'Type of model',
	`current_stage` VARCHAR(64) COMMENT 'Model stage ex: None, production...',
	`creation_time` DATETIME(3) COMMENT 'Millisecond precision',
	`last_updated_time` DATETIME(3) COMMENT 'Millisecond precision',
	`dataset` VARCHAR(256) COMMENT 'Which dataset is used',
	`description` VARCHAR(5000),
	CONSTRAINT `model_version_pk` PRIMARY KEY (`name`, `version`),
	FOREIGN KEY(`name`) REFERENCES `registered_model` (`name`) ON UPDATE CASCADE ON DELETE CASCADE,
	UNIQUE(`source`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

DROP TABLE IF EXISTS `model_version_tag`;
CREATE TABLE `model_version_tag` (
	`name` VARCHAR(256) NOT NULL COMMENT 'Name of model',
	`version` INTEGER NOT NULL,
	`tag` VARCHAR(256) NOT NULL,
	CONSTRAINT `model_version_tag_pk` PRIMARY KEY (`name`, `version`, `tag`),
	FOREIGN KEY(`name`, `version`) REFERENCES `model_version` (`name`, `version`) ON UPDATE CASCADE ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

DROP TABLE IF EXISTS `metric`;
CREATE TABLE `metric` (
	`id` VARCHAR(64) NOT NULL COMMENT 'Id of the Experiment',
	`key` VARCHAR(190) NOT NULL COMMENT 'Metric key: `String` (limit 190 characters). Part of *Primary Key* for ``metric`` table.',
	`value` FLOAT NOT NULL COMMENT 'Metric value: `Float`. Defined as *Non-null* in schema.',
	`worker_index` VARCHAR(32) NOT NULL COMMENT 'Metric worker_index: `String` (limit 32 characters). Part of *Primary Key* for\r\n    ``metrics`` table.',
	`timestamp` DATETIME(3) NOT NULL COMMENT 'Timestamp recorded for this metric entry: `DATETIME` (millisecond precision). 
											 Part of *Primary Key* for   ``metrics`` table.',
	`step` INTEGER NOT NULL COMMENT 'Step recorded for this metric entry: `INTEGER`.',
	`is_nan` BOOLEAN NOT NULL COMMENT 'True if the value is in fact NaN.',
	CONSTRAINT `metric_pk` PRIMARY KEY  (`id`, `key`, `timestamp`, `worker_index`),
	FOREIGN KEY(`id`) REFERENCES `experiment` (`id`) ON UPDATE CASCADE ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

DROP TABLE IF EXISTS `param`;
CREATE TABLE `param` (
	`id` VARCHAR(64) NOT NULL COMMENT 'Id of the Experiment',
	`key` VARCHAR(190) NOT NULL COMMENT '`String` (limit 190 characters). Part of *Primary Key* for ``param`` table.',
	`value` VARCHAR(190) NOT NULL COMMENT '`String` (limit 190 characters). Defined as *Non-null* in schema.',
	`worker_index` VARCHAR(32) NOT NULL COMMENT '`String` (limit 32 characters). Part of *Primary Key* for\r\n    ``metric`` table.',
	CONSTRAINT `param_pk` PRIMARY KEY  (`id`, `key`, `worker_index`),
	FOREIGN KEY(`id`) REFERENCES `experiment` (`id`) ON UPDATE CASCADE ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
