
! ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- |

SUBROUTINE eflux(uu)

    USE params
    USE vars
    USE phys
    USE fmt
    
    IMPLICIT NONE

    DOUBLE COMPLEX, DIMENSION(nn), INTENT(IN) :: uu

    INTEGER :: i

    DOUBLE PRECISION, DIMENSION(nn) :: flux

    flux(1) = IMAG( uu(1) * uu(2) * CONJG(uu(3)) ) * LAM
    
    DO i = 2, nn-2
        flux(i) = IMAG( uu(i  ) * uu(i+1) * CONJG(uu(i+2)) ) * LAM + &
                  IMAG( uu(i-1) * uu(i  ) * CONJG(uu(i+1)) ) / LAM
    END DO

    flux(nn-1) = IMAG( uu(nn-2) * uu(nn-1) * CONJG(uu(nn)) ) / LAM
    flux(nn  ) = 0.0D0
    
    OPEN(1, file='Results/flux.' // ext // '.txt', position='append')
    DO i = 1, nn
        WRITE(1,*) kk(i), -flux(i)*kk(i)
    ENDDO
    CLOSE(1)

END SUBROUTINE eflux

! ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- |

SUBROUTINE nonl(uu, nl)

    USE params
    USE vars
    USE phys
    
    IMPLICIT NONE
    
    DOUBLE COMPLEX, DIMENSION(nn), INTENT(IN)  :: uu
    DOUBLE COMPLEX, DIMENSION(nn), INTENT(OUT) :: nl

    INTEGER :: i

    nl(1) = kk(2) * AA * CONJG(uu(2)) * uu(3)
    nl(2) = kk(3) * AA * CONJG(uu(3)) * uu(4) + &
            kk(2) * BB * CONJG(uu(1)) * uu(3)
            
    DO i = 3, nn - 2

        nl(i) = kk(i+1) * AA * CONJG(uu(i+1)) * uu(i+2) + &
                kk(i  ) * BB * CONJG(uu(i-1)) * uu(i+1) - &
                kk(i-1) * CC *       uu(i-2)  * uu(i-1)
        
    END DO

    nl(nn-1) =   kk(nn-1) * BB * CONJG(uu(nn-2)) * uu(nn  ) - &
                 kk(nn-2) * CC *       uu(nn-3)  * uu(nn-2)
    nl(nn  ) = - kk(nn-1) * CC *       uu(nn-2)  * uu(nn-1)
    
END SUBROUTINE nonl
