import { MenuFoldOutlined, UserOutlined } from "@ant-design/icons";
import "./Header.scss";

export default function Header(props: any) {
  return (
    <div className="header">
      <MenuFoldOutlined />
      <div className="profile">
        <UserOutlined />
        <span className="user">Admin</span>
      </div>
    </div>
  );
}
