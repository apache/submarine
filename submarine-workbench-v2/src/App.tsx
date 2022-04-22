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

import Sidebar from "@submarine/components/Sidebar";
import { Navigate, Route, Routes } from "react-router-dom";
import Notebook from "@submarine/pages/workbench/notebook/Notebook";
import Experiment from "@submarine/pages/workbench/experiment/Experiment";
import Template from "@submarine/pages/workbench/template/Template";
import Environment from "@submarine/pages/workbench/environment/Environment";
import User from "@submarine/pages/workbench/user/User";
import DataDict from "@submarine/pages/workbench/data_dict/DataDict";
import Department from "@submarine/pages/workbench/department/Department";
import Model from "@submarine/pages/workbench/model/Model";
import { Layout } from "antd";
import { useState } from "react";

function App() {
  const [isCollapsed, setIsCollapsed] = useState(false);

  return (
    <Layout hasSider>
      <Sidebar
        {...{
          isCollapsed: isCollapsed,
          setIsCollapsed: setIsCollapsed,
        }}
      ></Sidebar>
      <Layout
        data-testid="page-layout"
        style={isCollapsed ? { paddingLeft: "80px", height: "100vh" } : { paddingLeft: "256px", height: "100vh" }}
      >
        <Routes>
          <Route path="/" element={<Navigate replace to="experiment" />} />
          <Route path="notebook" element={<Notebook />} />
          <Route path="experiment" element={<Experiment />} />
          <Route path="template" element={<Template />} />
          <Route path="environment" element={<Environment />} />
          <Route path="user" element={<User />} />
          <Route path="dataDict" element={<DataDict />} />
          <Route path="department" element={<Department />} />
          <Route path="model" element={<Model />} />
        </Routes>
      </Layout>
    </Layout>
  );
}

export default App;
