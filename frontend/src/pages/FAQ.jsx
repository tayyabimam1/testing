import React, { useState } from 'react'

const FAQ = () => {
  const [openQuestion, setOpenQuestion] = useState(null);

  const faqData = [
    {
      id: 1,
      question: 'How accurate is deepfake detection?',
      answer: 'Our deepfake detection system achieves high accuracy through advanced AI and machine learning algorithms. However, accuracy can vary depending on video quality and sophistication of the deepfake.'
    },
    {
      id: 2,
      question: 'Can deepfakes be created from just one photo?',
      answer: 'Yes, modern deepfake technology can create fake videos from a single photo. However, the quality and realism typically improve with more source material.'
    },
    {
      id: 3,
      question: 'Are there legitimate uses for synthetic media technology?',
      answer: 'Yes, synthetic media has many positive applications in entertainment, education, and business. It is used in movies, video games, virtual training, and accessibility tools.'
    },
    {
      id: 4,
      question: 'How can I protect myself from deepfakes?',
      answer: 'Protect yourself by verifying sources, being cautious of unusual video content, and using detection tools like ours. Digital literacy and awareness are key defenses.'
    },
    {
      id: 5,
      question: 'Will deepfakes eventually become undetectable?',
      answer: 'While deepfake technology continues to advance, detection methods are also evolving. We continuously update our AI models to keep pace with new deepfake techniques.'
    }
  ];

  const toggleQuestion = (id) => {
    setOpenQuestion(openQuestion === id ? null : id);
  };

  return (
    <div className="custom-bg min-vh-100 py-5">
      <div className="container py-5">
        <h1 className="text-color text-center display-5 fw-bold mb-2">Frequently Asked Questions</h1>
        <p className="text-center text-mute fs-5 mb-5">Get answers to common questions about deepfake detection</p>
        
        <div className="faq-container mx-auto" style={{ maxWidth: '800px' }}>
          {faqData.map((faq) => (
            <div key={faq.id} className="faq-item mb-3">
              <button
                className={`faq-question w-100 text-start border-0 p-4 rounded ${openQuestion === faq.id ? 'active' : ''}`}
                onClick={() => toggleQuestion(faq.id)}
                style={{
                  background: 'rgba(255, 255, 255, 0.05)',
                  color: '#94A3B8',
                  transition: 'all 0.3s ease'
                }}
              >
                <div className="d-flex justify-content-between align-items-center">
                  <span>{faq.question}</span>
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    width="24"
                    height="24"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    style={{
                      transform: openQuestion === faq.id ? 'rotate(180deg)' : 'rotate(0)',
                      transition: 'transform 0.3s ease'
                    }}
                  >
                    <polyline points="6 9 12 15 18 9"></polyline>
                  </svg>
                </div>
              </button>
              {openQuestion === faq.id && (
                <div 
                  className="faq-answer p-4"
                  style={{
                    background: 'rgba(255, 255, 255, 0.02)',
                    color: '#94A3B8',
                    borderBottomLeftRadius: '0.375rem',
                    borderBottomRightRadius: '0.375rem'
                  }}
                >
                  {faq.answer}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

export default FAQ