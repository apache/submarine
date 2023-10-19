import { Breadcrumb } from "antd";
import "./TabHeader.scss";
export default function TabHeader() {
  return (
    <div className="tab_header">
      <div className="accordion">
        <Breadcrumb>
          <Breadcrumb.Item>Home</Breadcrumb.Item>
          <Breadcrumb.Item>
            <a href="">Experiment</a>
          </Breadcrumb.Item>
        </Breadcrumb>
      </div>
      <div className="page">Experiment</div>
    </div>
  );
}
