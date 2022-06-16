/*!
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

import Header from "@submarine/components/Header";
import TabHeader from "@submarine/components/TabHeader";
import { Layout, Space, Tag } from "antd";
import { Radio, Table } from "antd";
import type { ColumnsType } from "antd/lib/table";
import { Action } from "history";
import React, { useState } from "react";
import "./Experiment.scss";

const { Content } = Layout;

//Action on the each table item
interface Actions {
  clone: string;
  update: string;
  delete: string;
}

//Keys in the table
interface DataType {
  key: React.Key;
  name: string;
  Id: string;
  tags: string[];
  status: string;
  finishedTime: string;
  createdTime: string;
  runningTime: string;
  duration: string;
  action: string;
}

//Data Model

const columns: ColumnsType<DataType> = [
  {
    title: "Experiment Name",
    dataIndex: "name",
    render: (text: string) => <a>{text}</a>,
  },
  {
    title: "Experiment ID",
    dataIndex: "Id",
    render: (text: string) => <a>{text}</a>,
  },
  {
    title: "Tags",
    dataIndex: "tags",
    render: (_, { tags }) => (
      <>
        {tags.map((tag) => {
          let color = tag.length > 5 ? "geekblue" : "green";
          if (tag === "Preview") {
            color = "volcano";
          }
          return (
            <Tag color={color} key={tag}>
              {tag.toUpperCase()}
            </Tag>
          );
        })}
      </>
    ),
  },
  {
    title: "Status",
    dataIndex: "status",
  },
  {
    title: "Created Time",
    dataIndex: "createdTime",
  },
  {
    title: "Running Time",
    dataIndex: "runningTime",
  },
  {
    title: "Finish Time",
    dataIndex: "finishedTime",
  },
  {
    title: "Duration",
    dataIndex: "duration",
  },
  {
    title: "Action",
    dataIndex: "action",
    render: (_, record) => (
      <Space size="middle">
        <a>Clone</a>
        <a>Update</a>
        <a>Delete</a>
      </Space>
    ),
  },
];

const data: DataType[] = [
  {
    key: "1",
    name: "Tracking Example",
    Id: "32",
    createdTime: "10:00AM",
    tags: ["Preview", "machine"],
    status: "Done",
    finishedTime: "10:00AM",
    runningTime: "10:00AM",
    duration: "1",
    action: "Clone",
  },
  {
    key: "2",
    name: "Tracking Example",
    Id: "32",
    createdTime: "10:00AM",
    tags: ["Preview", "machine"],
    status: "Done",
    finishedTime: "10:00AM",
    runningTime: "10:00AM",
    duration: "1",
    action: "Clone",
  },
  {
    key: "3",
    name: "Tracking Example",
    Id: "32",
    createdTime: "10:00AM",
    tags: ["Preview", "machine"],
    status: "Done",
    finishedTime: "10:00AM",
    runningTime: "10:00AM",
    duration: "1",
    action: "Clone",
  },
  {
    key: "4",
    name: "Tracking Example",
    Id: "32",
    createdTime: "10:00AM",
    tags: ["Preview", "machine"],
    status: "Done",
    finishedTime: "10:00AM",
    runningTime: "10:00AM",
    duration: "1",
    action: "Clone",
  },
  {
    key: "5",
    name: "Tracking Example",
    Id: "32",
    createdTime: "10:00AM",
    tags: ["Preview", "machine"],
    status: "Done",
    finishedTime: "10:00AM",
    runningTime: "10:00AM",
    duration: "1",
    action: "Clone",
  },
];

// rowSelection object indicates the need for row selection
const rowSelection = {
  onChange: (selectedRowKeys: React.Key[], selectedRows: DataType[]) => {
    console.log(`selectedRowKeys: ${selectedRowKeys}`, "selectedRows: ", selectedRows);
  },
  getCheckboxProps: (record: DataType) => ({
    disabled: record.name === "Disabled User", // Column configuration not to be checked
    name: record.name,
  }),
};

function Experiment() {
  return (
    <Layout data-testid="experiment-page">
      <Content>
        <Header />
        <TabHeader />

        <div className="table">
          <div className="actions"></div>

          <Table
            rowSelection={{
              ...rowSelection,
            }}
            columns={columns}
            dataSource={data}
          />
        </div>
      </Content>
    </Layout>
  );
}
export default Experiment;
