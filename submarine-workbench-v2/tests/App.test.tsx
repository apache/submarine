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

import { MemoryRouter } from "react-router-dom";
import { assert, describe, it } from "vitest";
import App from "../src/App";
import { render, cleanup, fireEvent } from "./utils/test-utils";

function renderPage() {
  return render(
    <MemoryRouter initialEntries={["/"]}>
      <App />
    </MemoryRouter>
  );
}

function testRouter(page: String) {
  const { getByTestId } = renderPage();
  fireEvent.click(getByTestId(`${page}-link`));
  assert.strictEqual(getByTestId(`${page}-item`).className.includes("ant-menu-item-selected"), true);
  expect(getByTestId(`${page}-page`)).toBeInTheDocument();
}

function testSubItemRouter(page: String) {
  const { getByTestId, getByText } = renderPage();
  fireEvent.click(getByText("Manager"));
  fireEvent.click(getByTestId(`${page}-link`));
  assert.strictEqual(getByTestId(`${page}-item`).className.includes("ant-menu-item-selected"), true);
  expect(getByTestId(`${page}-page`)).toBeInTheDocument();
}

describe("Router test", () => {
  it("Default start router", async () => {
    const { getByTestId, getByText } = render(
      <MemoryRouter initialEntries={["/"]}>
        <App />
      </MemoryRouter>
    );
    expect(getByTestId("experiment-page")).toBeInTheDocument();
  });

  it("Logo click", () => {
    const { getByTestId } = renderPage();
    fireEvent.click(getByTestId("logo"));
    expect(getByTestId("experiment-page")).toBeInTheDocument();
  });

  it("DataDict page router", () => {
    testSubItemRouter("data_dict");
  });

  it("Department page router", () => {
    testSubItemRouter("department");
  });

  it("Environment page router", () => {
    testRouter("environment");
  });

  it("Experiment page router", () => {
    testRouter("experiment");
  });

  it("Model page router", () => {
    testRouter("model");
  });

  it("Notebook page router", () => {
    testRouter("notebook");
  });

  it("Template page router", () => {
    testRouter("template");
  });

  it("User page router", () => {
    testSubItemRouter("user");
  });

  afterEach(cleanup);
});

describe("Collapsible test", () => {
  it("Collapsible test", () => {
    const { getByTestId } = renderPage();
    const experimentPage = getByTestId("page-layout");
    const sidebar = getByTestId("sidebar");
    expect(getComputedStyle(experimentPage).paddingLeft).toBe("256px");
    expect(getComputedStyle(sidebar).width).toBe("256px");
    const collapseElem = document.getElementsByClassName("ant-layout-sider-trigger")[0];
    fireEvent.click(collapseElem);
    expect(getComputedStyle(experimentPage).paddingLeft).toBe("80px");
    expect(getComputedStyle(sidebar).width).toBe("80px");
  });

  afterEach(cleanup);
});
