import React from "react";
import Button from "react-bootstrap/Button";
import Icon from "../assets/download-icon.png";
import Tick from "../assets/tick.png";
import Extension from "../assets/extension.jpg";
import { Container, Row, Col, Badge } from "react-bootstrap";
import { Link } from "react-router-dom";

const ExtensionPage = () => {
  const steps = [
    {
      title: "Download the Extension",
      description:
        "Download and install the DeepSight extension from github or google drive.",
    },
    {
      title: "Create an Account",
      description:
        "(Optional) Sign up for a free DeepSight account to access all features and track your analysis history.",
    },
    {
      title: "Load up Social Media Web Page",
      description:
        "Go to a social media web page such as instagram, twitter and Youtube load the extension from the extension section of the Chrome Browser.",
    },
    {
      title: "Analyze Content",
      description:
        'Once the extension opens up and the frame of the video you are trying to detect loads up press the detect button to anaylze.',
    },
    {
      title: "View Results",
      description:
        "Review the analysis results showing manipulation along with the confidence score.",
    },
  ];

  return (
    <div style={{ backgroundColor: "#0F1729" }}>
      <div
        className="container d-flex flex-column align-items-center justify-content-center"
        style={{ padding: "200px 0" }}
      >
        <div className="px-3 py-3 bg-primary rounded-circle d-inline-block">
          <img src={Icon} alt="image failed to load" />
        </div>
        <p className="display-4 fw-bold text-color">
          DeepSight Chrome Extension
        </p>
        <p className="text-mute fs-5">
          Detect deepfakes directly in your browser while browsing social media
        </p>
        <Button className="py-2 px-5 text-dark">
          <a target="_blank" rel="noopener noreferrer" href="https://drive.google.com/drive/folders/1nz8fbP6SZDpqRGEiQACvBxXhTetnhXo-?usp=sharing" className="text-decoration-none text-dark">
            Download Now &rArr;
          </a>
        </Button>
      </div>

      <div className="container d-flex flex-column flex-md-row justify-content-around align-items-center">
        <div className="col-md-6 col-12">
          <h1 className="text-color fw-bold">Detect Deepfake While Browsing</h1>
          <p className="text-mute">
            Our browser extension brings the power of DeepDetect directly to
            your browser, allowing you to analyze videos and images as you
            browse the web. With a simple right-click or toolbar button, you can
            instantly check if media content has been manipulated.
          </p>
          <ol className="list-unstyled">
            <li className="text-white">
              {" "}
              <img src={Tick} height={20} alt="" />{" "}
              <span className="fw-bold">Instant Analysis:</span> Check videos
              with a single click
            </li>
            <li className="text-white">
              {" "}
              <img src={Tick} height={20} alt="" />{" "}
              <span className="fw-bold">Social Media Integration:</span> Works
              on popular platforms like Instagram, Twitter, and{" "}
              <span className="ms-4">YouTube</span>
            </li>
            <li className="text-white">
              {" "}
              <img src={Tick} height={20} alt="" />{" "}
              <span className="fw-bold">Privacy-Focused:</span> All processing
              happens locally or through secure channels
            </li>
            <li className="text-white">
              {" "}
              <img src={Tick} height={20} alt="" />{" "}
              <span className="fw-bold">Real-time Alerts:</span> Get notified
              when potential deepfakes are detected
            </li>
          </ol>
        </div>

        <div className="col-md-6 col-12 d-flex justify-content-center">
          <img src={Extension} alt="" />
        </div>
      </div>

      {/* How to Use Section */}
      <div style={{padding: '100px 0'}}>
        <Container>
          <Row className="justify-content-center text-center mb-5">
            <Col lg={8}>
              <h2 className="display-4 fw-bold text-color">
                How to Use
              </h2>
              <p className="lead text-mute fs-5 lh-lg">
                Get started with the DeepDetect browser extension in just a few
                simple steps
              </p>
            </Col>
          </Row>

          <Row className="justify-content-center">
            <Col xl={8} lg={10}>
              {steps.map((step, index) => (
                <div
                  key={index}
                  className="d-flex align-items-start mb-4 position-relative"
                >
                  <div className="me-4 d-flex flex-column align-items-center">
                    <Badge
                      bg="primary"
                      className="rounded-circle d-flex align-items-center justify-content-center fw-bold"
                      style={{
                        width: "48px",
                        height: "48px",
                        fontSize: "18px",
                      }}
                    >
                      {index + 1}
                    </Badge>
                  </div>
                  <div className="flex-grow-1">
                    <h4 className="fw-semibold mb-2 text-white">{step.title}</h4>
                    <p className="text-mute mb-0 lh-lg">{step.description}</p>
                  </div>
                </div>
              ))}
            </Col>
          </Row>
        </Container>
      </div>

      <div className="text-center" style={{padding: '100px 0'}}>
        <h1 className="text-color fw-bold">Start Detecting Deepfakes Today</h1>
        <p className="text-mute">Download the DeepDetect browser extension and protect yourself from manipulated media</p>
        <Button className="py-2 mt-2 text-dark"><a className="text-decoration-none text-dark" href="https://drive.google.com/drive/folders/1nz8fbP6SZDpqRGEiQACvBxXhTetnhXo-?usp=sharing" target="_blank" rel="noopener noreferrer">Download Extension &rarr;</a></Button>
        <Button variant="outline-primary" className="py-2 mt-2 ms-3 text-white"><Link to='/predict' className='text-decoration-none text-white'>Try Web Version &rarr;</Link></Button>
      </div>

    </div>
  );
};

export default ExtensionPage;
