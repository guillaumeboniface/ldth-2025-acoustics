import Link from "next/link";
import Layout from "../components/Layout";
import Recorder from "../components/Recorder";
const IndexPage = () => (
  <Layout title="Home | Next.js + TypeScript Example">
    <Recorder />
  </Layout>
);

export default IndexPage;
