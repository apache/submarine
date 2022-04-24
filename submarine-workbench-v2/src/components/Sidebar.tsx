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

import { useState } from "react";
import { Layout, Menu } from "antd";
import {
  HomeOutlined,
  ExperimentOutlined,
  BookOutlined,
  FileOutlined,
  CodepenOutlined,
  InboxOutlined,
  SettingOutlined,
  BarChartOutlined,
  DesktopOutlined,
  ApiOutlined,
} from "@ant-design/icons";
import "./Sidebar.scss";
import { Link } from "react-router-dom";

const { Sider } = Layout;

function Sidebar(props: any) {
  const [key, setKey] = useState("experiment");

  const handleClick = (e: { key: string }) => {
    setKey(() => e.key);
  };

  const { isCollapsed, setIsCollapsed } = props;
  return (
    <Sider
      data-testid="sidebar"
      className="menu-sidebar"
      width="256px"
      breakpoint="md"
      collapsible
      onCollapse={(isCollapsed) => setIsCollapsed(isCollapsed)}
      collapsed={isCollapsed}
    >
      <Link data-testid="logo" to="/experiment" onClick={() => setKey("experiment")}>
        <div className="sidebar-logo">
          <img className="sidebar-logo-img" src="/logo.png" alt="logo" />
          <h1>Submarine</h1>
        </div>
      </Link>
      <div style={{ height: "calc(100% - 64px)", overflow: "overlay" }}>
        <Menu
          data-testid="menu"
          onClick={handleClick}
          theme="dark"
          defaultOpenKeys={["sub1"]}
          selectedKeys={[key]}
          mode="inline"
          className="menu"
        >
          <Menu.Item className="menu-item" key="home" icon={<HomeOutlined />} disabled={true}>
            <Link to="/home">Home</Link>
          </Menu.Item>
          <Menu.Item data-testid="notebook-item" key="notebook" icon={<BookOutlined />}>
            <Link data-testid="notebook-link" to="/notebook">
              Notebook
            </Link>
          </Menu.Item>
          <Menu.Item data-testid="experiment-item" key="experiment" icon={<ExperimentOutlined />}>
            <Link data-testid="experiment-link" to="/experiment">
              Experiment
            </Link>
          </Menu.Item>
          <Menu.Item data-testid="template-item" key="template" icon={<FileOutlined />}>
            <Link data-testid="template-link" to="/template">
              Template
            </Link>
          </Menu.Item>
          <Menu.Item data-testid="environment-item" key="environment" icon={<CodepenOutlined />}>
            <Link data-testid="environment-link" to="/environment">
              Environment
            </Link>
          </Menu.Item>
          <Menu.SubMenu data-testid="submenu" key="sub2" icon={<SettingOutlined />} title="Manager">
            <Menu.Item data-testid="user-item" key="user">
              <Link data-testid="user-link" to="/user">
                User
              </Link>
            </Menu.Item>
            <Menu.Item data-testid="data_dict-item" key="dataDict">
              <Link data-testid="data_dict-link" to="/dataDict">
                Data Dict
              </Link>
            </Menu.Item>
            <Menu.Item data-testid="department-item" key="department">
              <Link data-testid="department-link" to="/department">
                Department
              </Link>
            </Menu.Item>
          </Menu.SubMenu>
          <Menu.Item data-testid="model-item" key="model" icon={<InboxOutlined />}>
            <Link data-testid="model-link" to="/model">
              Model
            </Link>
          </Menu.Item>
          <Menu.Item data-testid="data-item" key="data" icon={<BarChartOutlined />} disabled={true}>
            <Link data-testid="data-link" to="/data">
              Data
            </Link>
          </Menu.Item>
          <Menu.Item data-testid="workspace-item" key="workspace" icon={<DesktopOutlined />} disabled={true}>
            <Link data-testid="workspace-link" to="/workspace">
              Workspace
            </Link>
          </Menu.Item>
          <Menu.Item data-testid="interpreter-item" key="interpreter" icon={<ApiOutlined />} disabled={true}>
            <Link data-testid="interpreter-link" to="/interpreter">
              Interpreter
            </Link>
          </Menu.Item>
        </Menu>
      </div>
    </Sider>
  );
}
export default Sidebar;
