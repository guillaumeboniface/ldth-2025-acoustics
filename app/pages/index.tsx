import Link from "next/link";
import Layout from "../src/components/Layout";
import Recorder from "../src/components/Recorder";
import { useEffect } from "react";
import init from "rust-melspec-wasm";

const IndexPage = () => {
  useEffect(() => {
    init().then(() => {
      console.log("Melspec initialized");
    });
  }, []);

  return (
    <Layout title="Home | Next.js + TypeScript Example">
      <Recorder />
    </Layout>
  );
};

export default IndexPage;
