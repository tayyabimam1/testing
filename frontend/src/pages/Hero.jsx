import React from 'react'
import { Link } from 'react-router-dom'
import Button from 'react-bootstrap/Button'
import Img1 from '@/assets/img1.jpg' 

const Hero = () => {
  return (
    <div className='custom-bg'>
      <div className="hero-section container-fluid px-4" style={{padding: '200px 0'}}>
        <div className="row align-items-center">
          <div className="col-md-6">
            <p className='display-4 fw-bold text-color'>Detect DeepFakes With Ease</p>
            <p className='text-mute fs-5'>Our advanced AI technology helps you identify manipulated videos with industry-leading accuracy.</p>
            <Button variant="primary" className='text-dark px-5 fw-bold mt-2'><Link to='/predict' className='text-decoration-none fw-bold text-dark'>Try it &rarr;</Link></Button>
            <Button variant='outline-primary' className='text-white px-5 fw-bold mt-2 ms-3'><a href='#work' className='text-decoration-none fw-bold text-white'>How it works</a></Button>
          </div>
          <div className="col-md-6 text-center">
            <img src={Img1} alt="Failed to load image" className='img-fluid' style={{maxHeight: '400px'}} />
          </div>
        </div>
      </div>

      <div id="work" className="card-contain-color" style={{padding: '100px 0'}}>
        <h1 className='text-color text-center'>How It Works</h1>
        <p className='text-center text-mute fs-5'>Our advanced AI technology analyzes videos frame by frame to detect manipulations</p>
        <div className="py-5 d-flex align-items-center justify-content-center container">
          <div className="col-md-4 text-center card1">
            <span className="px-3 py-3 bg-primary rounded-circle d-inline-block">
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-video h-8 w-8 text-blue-600 dark:text-blue-300 fs-4 text-color"><path d="m16 13 5.223 3.482a.5.5 0 0 0 .777-.416V7.87a.5.5 0 0 0-.752-.432L16 10.5"></path><rect x="2" y="6" width="14" height="12" rx="2"></rect></svg>
            </span>
            <p className='text-white fw-bold fs-4'>Upload Video</p>
            <p className='text-mute text-center' style={{marginTop: '-15px'}}>Upload any video file for analysis using our secure drag and drop interface</p>
          </div>

          <div className="col-md-4 text-center card2">
            <span className="px-3 py-3 bg-primary rounded-circle d-inline-block">
              <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" className="text-white fs-3 d-inline-block"><path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"></path></svg>
            </span>
            <p className='text-white fw-bold fs-4'>AI Analysis</p>
            <p className='text-mute text-center' style={{marginTop: '-15px'}}>Our advanced AI model analyzes the video for signs of manipulation</p>
          </div>

          <div className="col-md-4 text-center card2">
            <span className="px-3 py-3 bg-primary rounded-circle d-inline-block">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-chart-column h-8 w-8 text-white"><path d="M3 3v16a2 2 0 0 0 2 2h16"></path><path d="M18 17V9"></path><path d="M13 17V5"></path><path d="M8 17v-3"></path></svg>
            </span>
            <p className='text-white fw-bold fs-4'>Get Results</p>
            <p className='text-mute text-center' style={{marginTop: '-15px'}}>Receive detailed analysis with confidence scores and visual indicators</p>
          </div>

        </div>
      </div>
    </div>
  )
}

export default Hero